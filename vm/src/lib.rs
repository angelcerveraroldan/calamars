use calamars_core::{Identifier, ids};

use ir::{BinaryOperator, Consts, ValueId};

pub enum VmError {
    NotYetImplemented,
    DestinationIsNeeded,
    MissingTerminator,

    InternalFunctionIdNotFound,
    InternalBlockIdNotFound,
    InternalValueIdNotFound,
}

pub struct Register(usize);
pub type VmRes<A> = Result<A, VmError>;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Value {
    Integer(i64),
    Boolean(bool),
}

pub struct VirtualMachine {
    values: Vec<Value>,
    types: Vec<ids::TypeId>,
}

impl VirtualMachine {
    fn execude_instruction(&mut self, inst: Bytecode) -> VmRes<()> {
        Ok(())
    }
}

pub enum Bytecode {
    /// Save an integer to some register
    ConstI64(Register, i64),
    Add {
        a: Register,
        b: Register,
        /// Where to save the new value. This may be the same as a or b in the case of mutation.
        to: Register,
    },
    Ret(Register),
}

pub struct VmFunction {
    name: ids::IdentId,
    arity: u16,
    register_count: u16,
    bytecode: Vec<Bytecode>,
}

impl VmFunction {
    pub fn new(name: ids::IdentId, arity: u16, nreg: u16) -> Self {
        VmFunction {
            name,
            arity,
            register_count: nreg,
            bytecode: Vec::new(),
        }
    }
}

struct Lowerer<'a> {
    ctx: &'a ir::Module,
}

impl<'a> Lowerer<'a> {
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
            _ => Err(VmError::NotYetImplemented),
        }
    }

    fn lower_terminator(&self, term: &ir::Terminator) -> VmRes<Vec<Bytecode>> {
        match term {
            ir::Terminator::Return(ret_valueid) => ret_valueid
                .map(|vid| vec![Bytecode::Ret(self.register_dest(vid))])
                // Cannot return nothing yet
                .ok_or(VmError::NotYetImplemented),
            _ => Err(VmError::NotYetImplemented),
        }
    }

    fn lower_function_byid(&self, funcid: &ir::FunctionId) -> VmRes<VmFunction> {
        let f = self
            .ctx
            .functions
            .get(funcid.inner_id())
            .ok_or(VmError::InternalFunctionIdNotFound)?;

        self.lower_function(f)
    }

    fn lower_function(&self, fun: &ir::Function) -> VmRes<VmFunction> {
        if fun.blocks.len() != 1 {
            return Err(VmError::NotYetImplemented);
        }
        let entry_block = fun.blocks.first().ok_or(VmError::InternalBlockIdNotFound)?;
        todo!()
    }
}
