//! A virtual machine to run Calamars
//!
//! For now, this file is meant to "just sort of work", and thus it is VERY inneficient. Once the
//! project reaches a state where the front end is semi-stable, and a stdlib exists, much
//! optimization needs to be done here.

use calamars_core::{Identifier, ids};

use ir::{BinaryOperator, BlockId, Consts, FunctionId, Module, ValueId, lower::MirRes};

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
    CannotCallPhiOnFirstBlock,
    PhiFailedDidNotFindLastBranch,
    MainFunctionMustReturnInt,
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
    functions: Vec<VmFunction>,
    entry_point: FunctionId,
}

impl VirtualMachine {
    pub fn new(functions: Vec<VmFunction>, entry_point: FunctionId) -> Self {
        Self {
            functions,
            entry_point,
        }
    }

    pub fn run_function(&self, fid: FunctionId, inputs: Vec<Value>) -> VmRes<Value> {
        self.functions[fid.inner_id()]
            .runner(inputs)
            .run_function(self)
    }

    pub fn run_module(&self) -> VmRes<()> {
        let entry = self.entry_point;
        let entry_out = self.run_function(entry, vec![])?;
        match entry_out {
            Value::Integer(sysout) => {
                // std::process::exit(sysout as i32);
		println!("Exited with: {}", sysout);
		Ok(())
            }
            _ => Err(VmError::MainFunctionMustReturnInt),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinaryByteInfo {
    pub a: Register,
    pub b: Register,
    pub to: Register,
}

impl BinaryByteInfo {
    fn new(a: Register, b: Register, to: Register) -> Self {
        Self { a, b, to }
    }
}

#[derive(Clone, Debug)]
pub enum Bytecode {
    /// Save an integer to some register
    ConstI64(Register, i64),
    EqEq(BinaryByteInfo),
    NotEqual(BinaryByteInfo),
    Add(BinaryByteInfo),
    Sub(BinaryByteInfo),
    Lesser(BinaryByteInfo),
    Greater(BinaryByteInfo),
    Geq(BinaryByteInfo),
    Leq(BinaryByteInfo),
    Times(BinaryByteInfo),
    Div(BinaryByteInfo),
    Modulo(BinaryByteInfo),

    /// Jump to a different bytecode
    Br(BlockId),
    BrIf(Register, BlockId, BlockId),

    Phi(Register, Box<[(BlockId, Register)]>),
    Call(ir::FunctionId, Box<[ValueId]>, Register),

    /// Return the value in some register
    Ret(Register),
}

#[derive(Debug)]
pub struct VmFunction {
    name: ids::IdentId,
    fid: FunctionId,
    arity: u16,
    register_count: u16,

    block_ind: Vec<usize>,
    bytecode: Vec<Bytecode>,
}

impl VmFunction {
    pub fn new(
        name: ids::IdentId,
        fid: FunctionId,
        arity: u16,
        nreg: u16,
        bytecode: Vec<Bytecode>,
        block_ind: Vec<usize>,
    ) -> Self {
        VmFunction {
            name,
            fid,
            arity,
            register_count: nreg,
            bytecode,
            block_ind,
        }
    }

    fn runner(&self, mut inputs: Vec<Value>) -> VmFunctionRunner {
        // Opitimization:
        // - Dont malloc every time, instead re use registers from other functions
        // - Use pointers, dont clone
        let padding = vec![Value::Empty; self.register_count as usize - inputs.len()];
        inputs.extend(padding);
        let runner = VmFunctionRunner {
            current_block: BlockId::from(0),
            last_blocks: Vec::new(),
            registers: inputs,
            bytecode: self.bytecode.clone(),
            instruction_count: 0,
            block_ind: self.block_ind.clone(),
        };
        runner
    }
}

#[derive(Debug)]
pub struct VmFunctionRunner {
    registers: Vec<Value>,
    bytecode: Vec<Bytecode>,

    /// What was the block prior to this one - None if we are in the 0th block
    last_blocks: Vec<BlockId>,
    current_block: BlockId,
    /// A mapping from block number to first bytecode in that block
    ///
    /// Used for jumping between blocks
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

    fn jump_block(&mut self, to: BlockId) -> VmRes<()> {
        let inner = to.inner_id();
        self.last_blocks.push(self.current_block);
        self.current_block = to;
        self.instruction_count = self
            .block_ind
            .get(inner)
            .copied()
            .ok_or(VmError::InternalBlockIdNotFound)?;
        Ok(())
    }

    fn next_inst(&mut self) {
        self.instruction_count += 1;
    }

    fn run_set_const(&mut self, to: &Register, value: Value) -> VmRes<()> {
        let reg = self.get_register_mut(to)?;
        *reg = value;
        Ok(())
    }

    fn run_numeric(
        &mut self,
        to: &Register,
        lhs: &Register,
        rhs: &Register,
        f: fn(&i64, &i64) -> i64,
    ) -> VmRes<()> {
        let lhs = self.get_register(lhs)?;
        let rhs = self.get_register(rhs)?;

        match (lhs, rhs) {
            (Value::Integer(li), Value::Integer(ri)) => {
                self.run_set_const(to, Value::Integer(f(li, ri)))
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

    fn run_phi(&mut self, to: &Register, branches: &Box<[(BlockId, Register)]>) -> VmRes<()> {
        let last = self
            .last_blocks
            .pop()
            .ok_or(VmError::CannotCallPhiOnFirstBlock)?;

        for (b, r) in branches {
            if *b == last {
                let val = self.get_register(r)?.clone();
                let to = self.get_register_mut(to)?;
                *to = val;
                return Ok(());
            }
        }

        Err(VmError::PhiFailedDidNotFindLastBranch)
    }

    fn run_bytecode(&mut self, bytecode: &Bytecode, vm: &VirtualMachine) -> VmRes<Option<Value>> {
        match bytecode {
            Bytecode::ConstI64(register, i) => self
                .run_set_const(register, Value::Integer(*i))
                .map(|_| None),
            Bytecode::Add(BinaryByteInfo { a, b, to }) => {
                self.run_numeric(to, a, b, |x, y| *x + *y).map(|_| None)
            }
            Bytecode::Sub(BinaryByteInfo { a, b, to }) => {
                self.run_numeric(to, a, b, |x, y| *x - *y).map(|_| None)
            }
            Bytecode::Modulo(BinaryByteInfo { a, b, to }) => {
                self.run_numeric(to, a, b, |x, y| *x % *y).map(|_| None)
            }
            Bytecode::Times(BinaryByteInfo { a, b, to }) => {
                self.run_numeric(to, a, b, |x, y| *x * *y).map(|_| None)
            }
            Bytecode::Div(BinaryByteInfo { a, b, to }) => {
                self.run_numeric(to, a, b, |x, y| *x / *y).map(|_| None)
            }
            Bytecode::EqEq(BinaryByteInfo { a, b, to }) => self.run_eqeq(to, a, b).map(|_| None),
            Bytecode::NotEqual(BinaryByteInfo { a, b, to }) => self.run_neq(to, a, b).map(|_| None),
            // Fixme: For now we are cloning, but here we need to implement some semantic rules as to how
            // values are moved. This is just a "for now" sort of thing.
            Bytecode::Call(fid, params, to) => {
                let params = params
                    .iter()
                    .map(|vid| self.registers.get(vid.inner_id()).unwrap().clone())
                    .collect();
                let value = vm.run_function(*fid, params)?;
                self.run_set_const(to, value)?;
                Ok(None)
            }
            Bytecode::Ret(register) => self.get_register(register).map(|ptr| Some(ptr.clone())),
            Bytecode::Br(instruction) => Ok(None),
            Bytecode::BrIf(cond, then, otherwise) => Ok(None),
            Bytecode::Phi(to, branches) => self.run_phi(to, branches).map(|_| None),
            _ => {
                return Err(VmError::NotYetImplemented(
                    "This binary op is not yet supported".to_string(),
                ));
            }
        }
    }

    fn run_function(&mut self, vm: &VirtualMachine) -> VmRes<Value> {
        while self.instruction_count < self.bytecode.len() {
            let bc = self.bytecode[self.instruction_count].clone();
            if let Some(value) = self.run_bytecode(&bc, vm)? {
                return Ok(value);
            }
            match bc {
                Bytecode::Br(to) => self.jump_block(to)?,
                Bytecode::BrIf(cond, then, otherwise) => match self.get_register(&cond)? {
                    Value::Boolean(true) => self.jump_block(then)?,
                    Value::Boolean(false) => self.jump_block(otherwise)?,
                    _ => return Err(VmError::WrongType),
                },
                _ => self.next_inst(),
            };
        }
        // Functions must return something, for now.
        Err(VmError::MissingTerminator)
    }
}

/// Lower a function from mir to Bytecode
pub struct Lowerer<'a> {
    ctx: &'a ir::Module,
}

impl<'a> Lowerer<'a> {
    pub fn new(ctx: &'a ir::Module) -> Self {
        Self { ctx }
    }

    fn register_dest(&self, vid: ValueId) -> Register {
        Register(vid.inner_id())
    }

    fn lower_binary(
        &self,
        op: &BinaryOperator,
        lhs: &ValueId,
        rhs: &ValueId,

        to: Register,
    ) -> VmRes<Vec<Bytecode>> {
        let a = self.register_dest(*lhs);
        let b = self.register_dest(*rhs);
        let bbi = BinaryByteInfo { a, b, to };

        let r = match op {
            BinaryOperator::Add => Bytecode::Add(bbi),
            BinaryOperator::EqEq => Bytecode::EqEq(bbi),
            BinaryOperator::NotEqual => Bytecode::NotEqual(bbi),
            BinaryOperator::Lesser => Bytecode::Lesser(bbi),
            BinaryOperator::Leq => Bytecode::Leq(bbi),
            BinaryOperator::Greater => Bytecode::Greater(bbi),
            BinaryOperator::Geq => Bytecode::Geq(bbi),
            BinaryOperator::Sub => Bytecode::Sub(bbi),
            BinaryOperator::Times => Bytecode::Times(bbi),
            BinaryOperator::Div => Bytecode::Div(bbi),
            BinaryOperator::Modulo => Bytecode::Modulo(bbi),
        };

        Ok(vec![r])
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
            ir::VInstructionKind::Binary { op, lhs, rhs } => {
                self.lower_binary(op, lhs, rhs, destination)
            }
            ir::VInstructionKind::Phi { ty, incoming } => {
                let branches = incoming
                    .clone()
                    .into_iter()
                    .map(|(x, y)| (x, self.register_dest(y)))
                    .collect::<Vec<_>>();

                Ok(vec![Bytecode::Phi(
                    destination,
                    branches.into_boxed_slice(),
                )])
            }
            ir::VInstructionKind::Call {
                callee: ir::Callee::Function(fid),
                args,
                return_ty,
            } => {
                let fncall = Bytecode::Call(*fid, args.clone().into(), destination);
                Ok(vec![fncall])
            }
            ir::VInstructionKind::Parameter { index, ty } => {
                // Do we really have to do anything here ?
                //
                // I think this is just all in startup
                Ok(vec![])
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

    fn lower_function(&mut self, fun: &ir::Function, fid: FunctionId) -> VmRes<VmFunction> {
        // let entry_block = fun.blocks.first().ok_or(VmError::InternalBlockIdNotFound)?;
        // let bytecode = self.lower_block(&fun.instructions, entry_block)?;
        let mut bytecode = vec![];
        let mut blocks_toid = Vec::with_capacity(fun.blocks.len());
        for (_, block) in fun.blocks.iter().enumerate() {
            let mut block_bytecode = self.lower_block(&fun.instructions, block)?;
            blocks_toid.push(bytecode.len());
            bytecode.append(&mut block_bytecode);
        }
        let fun = VmFunction::new(
            fun.name,
            fid,
            fun.arity(),
            fun.instructions.len() as u16,
            bytecode,
            blocks_toid,
        );
        Ok(fun)
    }

    pub fn lower_module(&mut self) -> VmRes<Vec<VmFunction>> {
        let mut functions = vec![];
        for (fid, func) in self.ctx.function_arena.inner().iter().enumerate() {
            let func = self.lower_function(func, fid.into())?;
            functions.push(func);
        }
        Ok(functions)
    }

    /// A temporary function, just for testing.
    ///
    /// For now we assume that the file has a single function, and run it. When support for
    /// function calls is added, we will call the `main` function here.
    pub fn finish(&mut self) -> VmRes<VirtualMachine> {
        let functions = self.lower_module()?;
        Ok(VirtualMachine {
            functions,
            entry_point: FunctionId::from(0),
        })
    }
}
