use crate::{
    bytecode::{BinOp, Bytecode, UnOp},
    errors::{VError, VResult},
    values::Value,
    Register,
};
use calamars_core::Identifier;
use ir;

type BytecodeIndex = u16;

/// The non-mutable state of a function
pub struct VFunction {
    id: ir::FunctionId,
    arity: u8,
    bytecode: Box<[Bytecode]>,
    /// Block -> BytecodeIndex map
    ///
    /// bbi_map[i] ~ First bytecode instructio in the ith block
    bbi_map: Vec<BytecodeIndex>,
    register_size: u32,
}

impl VFunction {
    pub fn new(
        id: ir::FunctionId,
        arity: u8,
        bytecode: Box<[Bytecode]>,
        bbi_map: Vec<BytecodeIndex>,
        register_size: u32,
    ) -> Self {
        Self {
            id,
            arity,
            bytecode,
            bbi_map,
            register_size,
        }
    }

    fn get_bytecode(&self, nth: usize) -> Option<&Bytecode> {
        self.bytecode.get(nth)
    }

    pub fn register_size(&self) -> u32 {
        self.register_size
    }
}

/// The current and the last block we have been to
///
/// This feels a little nicer than having current: BlockId and last: Option<BlockId>,
/// as it encodes the fact that we only ever not have a last if we are in the initial block,
/// which cannot contain a phi instruction.
enum BlockInfo {
    Initial(ir::BlockId),
    NonInitial {
        current: ir::BlockId,
        last: ir::BlockId,
    },
}

impl Default for BlockInfo {
    fn default() -> Self {
        Self::Initial(ir::BlockId::from(0))
    }
}

impl BlockInfo {
    fn mut_advance(&mut self, to: ir::BlockId) {
        *self = match &self {
            BlockInfo::Initial(block_id) => BlockInfo::NonInitial {
                current: to,
                last: *block_id,
            },
            BlockInfo::NonInitial { current, .. } => BlockInfo::NonInitial {
                current: to,
                last: *current,
            },
        };
    }

    /// Get the last block we were in before reaching the current block. If we are
    /// in the fisrt block, then None will be returned.
    fn last(&self) -> Option<&ir::BlockId> {
        match self {
            BlockInfo::Initial(_) => None,
            BlockInfo::NonInitial { last, .. } => Some(last),
        }
    }
}

/// Communicate to the VM what state the frame is currently in
pub enum FrameOut {
    /// Ask the vm to call a function for us, and save its return to our dst register
    FunctionCallPls {
        fid: ir::FunctionId,
        args: Vec<Value>,
        dst: Register,
    },
    /// We are returning what we have saved in some register. At this point our registers
    /// may be deallocated.
    Return(Register),
}

/// This is the mutable state of a running function
pub struct Frame {
    /// Immutable state of this function
    pub function: ir::FunctionId,
    registers: Vec<Value>,
    /// Bytecode instruction counter
    bc: BytecodeIndex,
    block_info: BlockInfo,
}

impl Frame {
    // Think of how to optimize inputs to avoid allocating here, as well as when
    // passing in the arguments.
    pub fn new(function: ir::FunctionId, register_count: u32) -> Self {
        let registers = vec![Value::Empty; register_count as usize];
        let block_info = BlockInfo::default();
        Frame {
            registers,
            block_info,
            bc: 0,
            function,
        }
    }

    pub fn store_value(&mut self, value: Value, dst: &Register) -> VResult<()> {
        let v = self.registers.get_mut(dst.inner_id() as usize);
        if let Some(v) = v {
            *v = value;
            return Ok(());
        }

        let err = VError::RegisterOutOfBounds {
            index: dst.inner_id(),
            len: self.registers.len() as u32,
        };
        Err(err)
    }

    pub fn read_register(&self, from: &Register) -> VResult<&Value> {
        self.registers
            .get(from.inner_id() as usize)
            .ok_or(VError::RegisterOutOfBounds {
                index: from.inner_id(),
                len: self.registers.len() as u32,
            })
    }

    fn read_register_as_bool(&self, from: &Register) -> VResult<bool> {
        match self.read_register(from)? {
            Value::Boolean(b) => Ok(*b),
            v => Err(VError::InvalidConditionType {
                found: v.type_name(),
            }),
        }
    }

    fn next_instruction(&mut self) {
        self.bc += 1;
    }

    fn jump_block(
        &mut self,
        to: ir::BlockId,
        // For some given block, what is the first bytecode instruction in it
        block_instruction_map: &[BytecodeIndex],
    ) -> VResult<()> {
        let nbc = block_instruction_map
            .get(to.inner_id())
            .ok_or(VError::BlockNotFound {
                block: to.inner_id() as u32,
            })?;
        self.bc = *nbc;
        self.block_info.mut_advance(to);
        Ok(())
    }

    fn phi_join_values_to_dest(
        &mut self,
        phi: &Box<[(ir::BlockId, Register)]>,
        dst: &Register,
    ) -> VResult<()> {
        let last = self
            .block_info
            .last()
            .ok_or(VError::PhiMissingIncoming { block: 0 })?;
        let (_, reg) =
            phi.iter()
                .find(|(blockid, _)| blockid == last)
                .ok_or(VError::PhiMissingIncoming {
                    block: last.inner_id() as u32,
                })?;
        let value = *self.read_register(reg)?;
        self.store_value(value, &dst)?;
        Ok(())
    }

    fn run_unary_instruct(&mut self, op: &UnOp, dst: &Register, a: &Register) -> VResult<()> {
        let v = self.read_register(a)?;
        let nv = match op {
            UnOp::Not | UnOp::Neg => v.negate(),
        }?;
        self.store_value(nv, dst)
    }

    fn run_binary_instruct(
        &mut self,
        op: &BinOp,
        dst: &Register,
        a: &Register,
        b: &Register,
    ) -> VResult<()> {
        let lhs = self.read_register(a)?;
        let rhs = self.read_register(b)?;
        let nval = match op {
            BinOp::Add => lhs.add(rhs),
            BinOp::Sub => lhs.sub(rhs),
            BinOp::Mul => lhs.mul(rhs),
            BinOp::Div => lhs.div(rhs),
            BinOp::Mod => lhs.modulus(rhs),
            BinOp::Eq => lhs.e(rhs),
            BinOp::Ne => lhs.e(rhs)?.negate(),
            BinOp::Lt => lhs.l(rhs),
            BinOp::Le => lhs.le(rhs),
            BinOp::Gt => lhs.g(rhs),
            BinOp::Ge => lhs.ge(rhs),
            BinOp::And => lhs.and(rhs),
            BinOp::Or => lhs.or(rhs),
            BinOp::Xor => lhs.xor(rhs),
        }?;
        self.store_value(nval, dst)?;
        Ok(())
    }

    /// Keep taking steps until we need the VM's help
    ///
    /// A frame for example, cannot handle `val x = foo()` by itsef, it needs
    /// to stop, and ask the vm to run `foo`.
    pub fn step_until_vm_is_needed(&mut self, vf: &VFunction) -> VResult<FrameOut> {
        let fo: FrameOut = loop {
            let bc = vf
                .get_bytecode(self.bc as usize)
                .ok_or(VError::BytecodeOutOfBounds {
                    index: self.bc as u32,
                    len: vf.bytecode.len() as u32,
                })?;
            match bc {
                // here we have no choice but to clone - we need the value to be stored in the frame, but it lives in the
                // vfunction. We dont want to move it from there, as it may be needed lated by another frame.
                Bytecode::Const { dst, k } => {
                    self.store_value(k.clone(), dst)?;
                    self.next_instruction();
                }
                Bytecode::Bin { op, dst, a, b } => {
                    self.run_binary_instruct(op, dst, a, b)?;
                    self.next_instruction();
                }
                Bytecode::Un { op, dst, x } => {
                    self.run_unary_instruct(op, dst, x)?;
                    self.next_instruction();
                }
                Bytecode::Br { target } => {
                    self.jump_block(*target, &vf.bbi_map)?;
                }
                Bytecode::BrIf {
                    cond,
                    then_t,
                    else_t,
                } => {
                    let b = self.read_register_as_bool(cond)?;
                    if b {
                        self.jump_block(*then_t, &vf.bbi_map)?;
                    } else {
                        self.jump_block(*else_t, &vf.bbi_map)?;
                    }
                }
                Bytecode::Call { callee, args, dst } => {
                    let args = args
                        .iter()
                        .map(|reg| self.read_register(reg).copied())
                        .collect::<VResult<Vec<_>>>()?;

                    // Here we are trusting that the VM will fill the register for us, when we are told to continue running,
                    // we can assume that the functions output will be in that register.
                    self.next_instruction();
                    break FrameOut::FunctionCallPls {
                        fid: *callee,
                        args: args,
                        dst: *dst,
                    };
                }
                Bytecode::Ret { src } => {
                    break FrameOut::Return(*src);
                }
                Bytecode::Phi { dst, incoming } => {
                    self.phi_join_values_to_dest(incoming, dst)?;
                    self.next_instruction();
                }
            };
        };
        Ok(fo)
    }

    /// Load inputs into the register
    pub fn load_inputs(&mut self, inputs: Vec<Value>) {
        self.registers[..inputs.len()].copy_from_slice(&inputs);
    }
}
