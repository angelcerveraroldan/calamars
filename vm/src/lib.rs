//! A virtual machine to run Calamars
//!
//! For now, this file is meant to "just sort of work", and thus it is VERY inneficient. Once the
//! project reaches a state where the front end is semi-stable, and a stdlib exists, much
//! optimization needs to be done here.

use std::{collections::HashMap, fmt::format};

use calamars_core::{Identifier, ids};

use ir::{BinaryOperator, BlockId, Consts, ValueId};

type BytecodeId = usize;

#[derive(Clone, Debug)]
pub enum VmError {
    NotYetImplemented(String),
    DestinationIsNeeded,
    MissingTerminator,

    CannotReadFromEmpyRegister,
    RegisterNotFound,
    MainFunctionNotFound,

    // Internal errors, these should not occur unless some invariant is broken
    InternalFunctionIdNotFound,
    InternalBlockIdNotFound,
    InternalValueIdNotFound,
    InternalInstructionNotFound,
    WrongType,
}

#[derive(Clone, Debug)]
pub struct Register(usize);

pub type VmRes<A> = Result<A, VmError>;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
/// A non-heap value
///
/// Later this will also contain pointers to potentially heap allocated values.
pub enum Value {
    Empty,

    Integer(i64),
    Boolean(bool),
}

impl Value {
    fn is_empty(&self) -> bool {
        matches!(self, Value::Empty)
    }
    fn is_full(&self) -> bool {
        !matches!(self, Value::Empty)
    }
}

pub struct VirtualMachine {
    values: Vec<Value>,
    types: Vec<ids::TypeId>,
}

impl VirtualMachine {
    fn execute_instruction(&mut self, inst: Bytecode) -> VmRes<()> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Bytecode {
    /// Save an integer to some register
    ConstI64(Register, i64),
    EqEq {
        a: Register,
        b: Register,
        to: Register,
    },
    NotEqual {
        a: Register,
        b: Register,
        to: Register,
    },
    Add {
        a: Register,
        b: Register,
        /// Where to save the new value. This may be the same as a or b in the case of mutation.
        to: Register,
    },

    /// Jump to a different bytecode
    Br(BlockId),
    BrIf(Register, BlockId, BlockId),
    /// Return the value in some register
    Ret(Register),
}

pub struct VmFunction {
    name: ids::IdentId,
    arity: u16,
    register_count: u16,

    block_ind: Vec<usize>,
    bytecode: Vec<Bytecode>,
}

impl VmFunction {
    pub fn new(
        name: ids::IdentId,
        arity: u16,
        nreg: u16,
        bytecode: Vec<Bytecode>,
        block_ind: Vec<usize>,
    ) -> Self {
        VmFunction {
            name,
            arity,
            register_count: nreg,
            bytecode,
            block_ind,
        }
    }

    fn runner(&self) -> VmFunctionRunner {
        // Opitimization:
        // - Dont malloc every time, instead re use registers from other functions
        // - Use pointers, dont clone
        VmFunctionRunner {
            registers: vec![Value::Empty; self.register_count as usize],
            bytecode: self.bytecode.clone(),
            instruction_count: 0,
            block_ind: self.block_ind.clone(),
        }
    }
}

pub struct VmFunctionRunner {
    registers: Vec<Value>,
    bytecode: Vec<Bytecode>,

    block_ind: Vec<usize>,
    instruction_count: usize,
}

impl VmFunctionRunner {
    fn get_register(&self, reg: &Register) -> VmRes<&Value> {
        self.registers.get(reg.0).ok_or(VmError::RegisterNotFound)
    }

    fn get_register_mut(&mut self, reg: &Register) -> VmRes<&mut Value> {
        self.registers
            .get_mut(reg.0)
            .ok_or(VmError::RegisterNotFound)
    }

    fn run_set_const(&mut self, to: &Register, value: Value) -> VmRes<()> {
        let reg = self.get_register_mut(to)?;
        *reg = value;
        Ok(())
    }

    fn run_add(&mut self, to: &Register, lhs: &Register, rhs: &Register) -> VmRes<()> {
        let lhs = self.get_register(lhs)?;
        let rhs = self.get_register(rhs)?;

        match (lhs, rhs) {
            (Value::Integer(li), Value::Integer(ri)) => {
                self.run_set_const(to, Value::Integer(*li + *ri))
            }
            // This should be unreachable ? Double check, and optimize for that.
            _ => Err(VmError::WrongType),
        }
    }

    fn run_eqeq(&mut self, to: &Register, lhs: &Register, rhs: &Register) -> VmRes<()> {
        let lhs = self.get_register(lhs)?;
        let rhs = self.get_register(rhs)?;

        match (lhs, rhs) {
            (Value::Integer(li), Value::Integer(ri)) => {
                self.run_set_const(to, Value::Boolean(li == ri))
            }
            (Value::Boolean(li), Value::Boolean(ri)) => {
                self.run_set_const(to, Value::Boolean(li == ri))
            }
            // This should be unreachable ? Double check, and optimize for that.
            _ => Err(VmError::WrongType),
        }
    }

    fn run_neq(&mut self, to: &Register, lhs: &Register, rhs: &Register) -> VmRes<()> {
        let lhs = self.get_register(lhs)?;
        let rhs = self.get_register(rhs)?;

        match (lhs, rhs) {
            (Value::Integer(li), Value::Integer(ri)) => {
                self.run_set_const(to, Value::Boolean(li != ri))
            }
            (Value::Boolean(li), Value::Boolean(ri)) => {
                self.run_set_const(to, Value::Boolean(li != ri))
            }
            // This should be unreachable ? Double check, and optimize for that.
            _ => Err(VmError::WrongType),
        }
    }

    fn run_bytecode(&mut self, bytecode: &Bytecode) -> VmRes<Option<Value>> {
        match bytecode {
            // Side effects
            Bytecode::ConstI64(register, i) => self
                .run_set_const(register, Value::Integer(*i))
                .map(|_| None),
            Bytecode::Add { a, b, to } => self.run_add(to, a, b).map(|_| None),
            Bytecode::EqEq { a, b, to } => self.run_eqeq(to, a, b).map(|_| None),
            Bytecode::NotEqual { a, b, to } => self.run_neq(to, a, b).map(|_| None),
            // Returns
            Bytecode::Ret(register) => self.get_register(register).map(|ptr| Some(ptr.clone())),
            Bytecode::Br(instruction) => Ok(None),
            Bytecode::BrIf(cond, then, otherwise) => Ok(None),
        }
    }

    fn run_function(&mut self) -> VmRes<Value> {
        while self.instruction_count < self.bytecode.len() {
            let bc = self.bytecode[self.instruction_count].clone();
            if let Some(value) = self.run_bytecode(&bc)? {
                return Ok(value);
            }
            self.instruction_count = match bc {
                Bytecode::Br(to) => self.block_ind[to.inner_id()],
                Bytecode::BrIf(cond, then, otherwise) => match self.get_register(&cond)? {
                    Value::Boolean(true) => self.block_ind[then.inner_id()],
                    Value::Boolean(false) => self.block_ind[otherwise.inner_id()],
                    _ => return Err(VmError::WrongType),
                },
                _ => self.instruction_count + 1,
            };
        }
        // Functions must return something, for now.
        Err(VmError::MissingTerminator)
    }
}

/// Lower a function from mir to Bytecode
pub struct Lowerer<'a> {
    ctx: &'a ir::Module,
    /// Block -> Index in bytecode for first instruction
    ///
    /// Used for block jumps
    blocks_toid: Vec<usize>,
}

impl<'a> Lowerer<'a> {
    pub fn new(ctx: &'a ir::Module) -> Self {
        Self {
            ctx,
            blocks_toid: Vec::new(),
        }
    }

    fn register_dest(&self, vid: ValueId) -> Register {
        Register(vid.inner_id())
    }

    fn lower_inst(
        &self,
        instruction: &ir::VInstruct,
        destination: Register,
    ) -> VmRes<Vec<Bytecode>> {
        match &instruction.kind {
            ir::VInstructionKind::Constant(Consts::I64(val)) => {
                Ok(vec![Bytecode::ConstI64(destination, *val)])
            }
            ir::VInstructionKind::Binary { op, lhs, rhs } if matches!(op, BinaryOperator::Add) => {
                Ok(vec![Bytecode::Add {
                    a: self.register_dest(*lhs),
                    b: self.register_dest(*rhs),
                    to: destination,
                }])
            }
            ir::VInstructionKind::Binary { op, lhs, rhs } if matches!(op, BinaryOperator::EqEq) => {
                Ok(vec![Bytecode::EqEq {
                    a: self.register_dest(*lhs),
                    b: self.register_dest(*rhs),
                    to: destination,
                }])
            }
            ir::VInstructionKind::Binary { op, lhs, rhs }
                if matches!(op, BinaryOperator::NotEqual) =>
            {
                Ok(vec![Bytecode::NotEqual {
                    a: self.register_dest(*lhs),
                    b: self.register_dest(*rhs),
                    to: destination,
                }])
            }
            inst => Err(VmError::NotYetImplemented(format!(
                "Instruction not supported: {:?}",
                inst
            ))),
        }
    }

    fn lower_terminator(&self, term: &ir::Terminator) -> VmRes<Vec<Bytecode>> {
        match term {
            ir::Terminator::Return(ret_valueid) => ret_valueid
                .map(|vid| vec![Bytecode::Ret(self.register_dest(vid))])
                // Cannot return nothing yet
                .ok_or(VmError::NotYetImplemented(
                    "Cannot return nothing".to_string(),
                )),
            ir::Terminator::Br { target } => Ok(vec![Bytecode::Br(*target)]),
            ir::Terminator::BrIf {
                condition,
                then_target,
                else_target,
            } => Ok(vec![Bytecode::BrIf(
                self.register_dest(*condition),
                *then_target,
                *else_target,
            )]),
            terminator => Err(VmError::NotYetImplemented(format!(
                "Terminator not supported: {:?}",
                terminator
            ))),
        }
    }

    fn lower_block(
        &self,
        instruction_pool: &Vec<ir::VInstruct>,
        block: &ir::BBlock,
    ) -> VmRes<Vec<Bytecode>> {
        let mut instructions = Vec::new();
        for inst_id in &block.instructs {
            let instruction = instruction_pool
                .get(inst_id.inner_id())
                .ok_or(VmError::InternalInstructionNotFound)?;
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

    fn lower_function_byid(&mut self, funcid: &ir::FunctionId) -> VmRes<VmFunction> {
        let f = self
            .ctx
            .functions
            .get(funcid.inner_id())
            .ok_or(VmError::InternalFunctionIdNotFound)?;

        self.lower_function(f)
    }

    fn lower_function(&mut self, fun: &ir::Function) -> VmRes<VmFunction> {
        // let entry_block = fun.blocks.first().ok_or(VmError::InternalBlockIdNotFound)?;
        // let bytecode = self.lower_block(&fun.instructions, entry_block)?;
        let mut bytecode = vec![];
        self.blocks_toid.reserve(fun.blocks.len());
        for (id, block) in fun.blocks.iter().enumerate() {
            let mut block_bytecode = self.lower_block(&fun.instructions, block)?;
            self.blocks_toid.push(bytecode.len());
            bytecode.append(&mut block_bytecode);
        }
        let fun = VmFunction::new(
            fun.name,
            fun.arity(),
            fun.instructions.len() as u16,
            bytecode,
            self.blocks_toid.clone(),
        );
        Ok(fun)
    }

    /// A temporary function, just for testing.
    ///
    /// For now we assume that the file has a single function, and run it. When support for
    /// function calls is added, we will call the `main` function here.
    pub fn run_module(&mut self) -> VmRes<Value> {
        let fun = self
            .ctx
            .functions
            .first()
            .ok_or(VmError::MainFunctionNotFound)?;
        let fun = self.lower_function(fun)?;
        let mut runner = fun.runner();
        runner.run_function()
    }
}
