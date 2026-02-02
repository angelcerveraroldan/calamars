use crate::{
    Register,
    bytecode::{BinOp, Bytecode, UnOp},
    errors::{VError, VResult},
    function::VFunction,
    values::Value,
};
use calamars_core::Identifier;
use ir::{self, FunctionId, ValueId};

fn callee_fnid(callee: &ir::Callee) -> VResult<FunctionId> {
    match callee {
        ir::Callee::Function(fid) => Ok(*fid),
        ir::Callee::Extern(_) => Err(VError::UnsupportedExtern),
    }
}

/// Lower MIR to VM bytecode
pub struct Lowerer<'a> {
    ctx: &'a ir::Module,
}

impl<'a> Lowerer<'a> {
    pub fn new(ctx: &'a ir::Module) -> Self {
        Self { ctx }
    }

    fn register_dest(&self, vid: ValueId) -> Register {
        Register::from(vid.inner_id() as u32)
    }

    fn lower_binary_op(&self, op: &ir::BinaryOperator) -> VResult<BinOp> {
        Ok(match op {
            ir::BinaryOperator::Add => BinOp::Add,
            ir::BinaryOperator::Sub => BinOp::Sub,
            ir::BinaryOperator::Times => BinOp::Mul,
            ir::BinaryOperator::Div => BinOp::Div,
            ir::BinaryOperator::Modulo => BinOp::Mod,
            ir::BinaryOperator::EqEq => BinOp::Eq,
            ir::BinaryOperator::NotEqual => BinOp::Ne,
            ir::BinaryOperator::Lesser => BinOp::Lt,
            ir::BinaryOperator::Leq => BinOp::Le,
            ir::BinaryOperator::Greater => BinOp::Gt,
            ir::BinaryOperator::Geq => BinOp::Ge,
            ir::BinaryOperator::And => BinOp::And,
            ir::BinaryOperator::Or => BinOp::Or,
            ir::BinaryOperator::Xor => BinOp::Xor,
        })
    }

    fn lower_bitwise_op(&self, op: &ir::BitwiseBinaryOperator) -> VResult<BinOp> {
        Ok(match op {
            ir::BitwiseBinaryOperator::And => BinOp::And,
            ir::BitwiseBinaryOperator::Or => BinOp::Or,
            ir::BitwiseBinaryOperator::Xor => BinOp::Xor,
        })
    }

    fn lower_unary_op(&self, op: &ir::UnaryOperator) -> VResult<UnOp> {
        Ok(match op {
            ir::UnaryOperator::Not => UnOp::Not,
            ir::UnaryOperator::Negate => UnOp::Neg,
        })
    }

    fn lower_inst(
        &self,
        instruction: &ir::VInstruct,
        destination: Register,
    ) -> VResult<Vec<Bytecode>> {
        match &instruction.kind {
            ir::VInstructionKind::Constant(constant) => {
                let k = match constant {
                    ir::Consts::I64(val) => Value::Integer(*val),
                    ir::Consts::Bool(val) => Value::Boolean(*val),
                    ir::Consts::Unit => return Err(VError::UnsupportedConstant),
                    ir::Consts::String(_) => return Err(VError::UnsupportedConstant),
                };
                Ok(vec![Bytecode::Const {
                    dst: destination,
                    k,
                }])
            }
            ir::VInstructionKind::Binary { op, lhs, rhs } => {
                let a = self.register_dest(*lhs);
                let b = self.register_dest(*rhs);
                let op = self.lower_binary_op(op)?;
                Ok(vec![Bytecode::Bin {
                    op,
                    dst: destination,
                    a,
                    b,
                }])
            }
            ir::VInstructionKind::BitwiseBinary { op, lhs, rhs } => {
                let a = self.register_dest(*lhs);
                let b = self.register_dest(*rhs);
                let op = self.lower_bitwise_op(op)?;
                Ok(vec![Bytecode::Bin {
                    op,
                    dst: destination,
                    a,
                    b,
                }])
            }
            ir::VInstructionKind::Unary { op, on } => {
                let x = self.register_dest(*on);
                let op = self.lower_unary_op(op)?;
                Ok(vec![Bytecode::Un {
                    op,
                    dst: destination,
                    x,
                }])
            }
            ir::VInstructionKind::Phi { incoming, .. } => {
                let branches = incoming
                    .iter()
                    .map(|(block, vid)| (*block, self.register_dest(*vid)))
                    .collect::<Vec<_>>();

                Ok(vec![Bytecode::Phi {
                    dst: destination,
                    incoming: branches.into_boxed_slice(),
                }])
            }
            ir::VInstructionKind::Call { callee, args, .. } => {
                let callee = callee_fnid(callee)?;
                let regs = args
                    .iter()
                    .map(|vid| self.register_dest(*vid))
                    .collect::<Vec<_>>();

                Ok(vec![Bytecode::Call {
                    callee,
                    args: regs.into_boxed_slice(),
                    dst: destination,
                }])
            }
            ir::VInstructionKind::Parameter { .. } => Ok(vec![]),
            ir::VInstructionKind::ConstDataPointer { .. } => Err(VError::UnsupportedInstruction),
        }
    }

    fn lower_terminator(&self, term: &ir::Terminator) -> VResult<Vec<Bytecode>> {
        match term {
            ir::Terminator::Return(ret_valueid) => ret_valueid
                .map(|vid| {
                    vec![Bytecode::Ret {
                        src: self.register_dest(vid),
                    }]
                })
                .ok_or(VError::InvalidReturnValue),
            ir::Terminator::Call { callee, args, .. } => {
                let callee = callee_fnid(callee)?;
                let regs = args
                    .iter()
                    .map(|vid| self.register_dest(*vid))
                    .collect::<Vec<_>>();

                Ok(vec![Bytecode::RetCall {
                    callee,
                    args: regs.into_boxed_slice(),
                }])
            }
            ir::Terminator::Br { target } => Ok(vec![Bytecode::Br { target: *target }]),
            ir::Terminator::BrIf {
                condition,
                then_target,
                else_target,
            } => Ok(vec![Bytecode::BrIf {
                cond: self.register_dest(*condition),
                then_t: *then_target,
                else_t: *else_target,
            }]),
        }
    }

    fn lower_block(
        &self,
        instruction_pool: &[ir::VInstruct],
        block: &ir::BBlock,
    ) -> VResult<Vec<Bytecode>> {
        let mut instructions = Vec::new();
        for inst_id in &block.instructs {
            let instruction = instruction_pool
                .get(inst_id.inner_id())
                .ok_or(VError::InternalInstructionNotFound)?;
            let destination = self.register_dest(*inst_id);
            let mut bytecode = self.lower_inst(instruction, destination)?;
            instructions.append(&mut bytecode);
        }

        if let Some(term) = &block.finally {
            let mut term_bytecode = self.lower_terminator(term)?;
            instructions.append(&mut term_bytecode);
        }

        Ok(instructions)
    }

    fn lower_function(&self, fun: &ir::Function, fid: FunctionId) -> VResult<VFunction> {
        let mut bytecode = vec![];
        let mut bbi_map = Vec::with_capacity(fun.blocks.len());
        // this needs to be optimized ....
        let register_size = fun.instructions.len() as u32;

        for block in fun.blocks.iter() {
            bbi_map.push(bytecode.len() as u16);
            let mut block_bytecode = self.lower_block(&fun.instructions, block)?;
            bytecode.append(&mut block_bytecode);
        }

        Ok(VFunction::new(
            fid,
            fun.arity() as u8,
            bytecode.into_boxed_slice(),
            bbi_map,
            register_size,
        ))
    }

    pub fn lower_module(&self) -> VResult<Vec<VFunction>> {
        let mut functions = Vec::new();
        for (fid, func) in self.ctx.function_arena.inner().iter().enumerate() {
            let func = self.lower_function(func, fid.into())?;
            functions.push(func);
        }
        Ok(functions)
    }
}
