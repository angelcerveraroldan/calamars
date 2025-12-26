use calamars_core::ids;
use ir::{BinaryOperator, Consts};

pub enum VmError {
    NotYetImplemented,
    DestinationIsNeeded,
    MissingTerminator,

    InternalFunctionIdNotFound,
    InternalBlockIdNotFound,
    InternalValueIdNotFound,
}

pub type Register = usize;
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
    fn lower_inst(&self, instruction: &ir::VInstruct) -> VmRes<Vec<Bytecode>> {
        let destination = instruction.dst.ok_or(VmError::DestinationIsNeeded)?.inner();
        match &instruction.kind {
            ir::VInstructionKind::Constant(Consts::I64(val)) => {
                Ok(vec![Bytecode::ConstI64(destination, *val)])
            }
            ir::VInstructionKind::Binary { op, lhs, rhs } if matches!(op, BinaryOperator::Add) => {
                Ok(vec![Bytecode::Add {
                    a: lhs.inner(),
                    b: rhs.inner(),
                    to: destination,
                }])
            }
            _ => Err(VmError::NotYetImplemented),
        }
    }

    fn lower_terminator(&self, term: &ir::Terminator) -> VmRes<Vec<Bytecode>> {
        match term {
            ir::Terminator::Return(ret_valueid) => ret_valueid
                .map(|dst| vec![Bytecode::Ret(dst.inner())])
                // Cannot return nothing yet
                .ok_or(VmError::NotYetImplemented),
            _ => Err(VmError::NotYetImplemented),
        }
    }

    fn lower_block(&self, block: &ir::BBlock) -> VmRes<Vec<Bytecode>> {
        let mut v = vec![];

        for value_id in &block.instructs {
            let instruct = self
                .ctx
                .values
                .get(*value_id)
                .ok_or(VmError::InternalValueIdNotFound)?;

            let mut byte = self.lower_inst(instruct)?;
            v.append(&mut byte);
        }

        if let Some(term) = &block.finally {
            let mut tins = self.lower_terminator(term)?;
            v.append(&mut tins);
            Ok(v)
        } else {
            Err(VmError::MissingTerminator)
        }
    }

    fn lower_function_byid(&self, funcid: &ir::FunctionId) -> VmRes<VmFunction> {
        let f = self
            .ctx
            .functions
            .get(*funcid)
            .ok_or(VmError::InternalFunctionIdNotFound)?;

        self.lower_function(f)
    }

    fn lower_function(&self, fun: &ir::Function) -> VmRes<VmFunction> {
        if fun.blocks.len() != 1 {
            return Err(VmError::NotYetImplemented);
        }

        let entry_block = self
            .ctx
            .blocks
            .get(fun.entry)
            .ok_or(VmError::InternalBlockIdNotFound)?;

        let bytecode = self.lower_block(entry_block)?;
        todo!()
    }
}
