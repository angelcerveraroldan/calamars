//! A virtual machine to run Calamars
//!
//! For now, this file is meant to "just sort of work", and thus it is VERY inneficient. Once the
//! project reaches a state where the front end is semi-stable, and a stdlib exists, much
//! optimization needs to be done here.

use crate::{
    errors::{VError, VResult},
    function::{Frame, VFunction},
    values::Value,
};
use calamars_core::Identifier;

mod bytecode;
mod errors;
mod function;
pub mod lower;
mod values;

// A register
#[derive(Clone, Debug, Copy)]
pub struct Register(u32);

impl From<u32> for Register {
    fn from(value: u32) -> Self {
        Register(value)
    }
}

impl Register {
    pub fn inner_id(&self) -> u32 {
        self.0
    }
}

/// Virtual Machine
pub struct VMachine {
    functions: Box<[VFunction]>,
    memory: Vec<Register>,
    stack: Vec<Frame>,
}

fn generate_frame(id: ir::FunctionId, fns: &[VFunction]) -> VResult<Frame> {
    let f = fns.get(id.inner_id()).ok_or(VError::FunctionNotFound {
        id: id.inner_id() as u32,
    })?;
    let frame = Frame::new(id, f.register_size());
    Ok(frame)
}

impl VMachine {
    pub fn new(functions: Box<[VFunction]>, entry: ir::FunctionId) -> VResult<Self> {
        let mut stack = Vec::new();
        let entry_frame = generate_frame(entry, &functions)?;
        stack.push(entry_frame);
        Ok(Self {
            functions,
            stack,
            memory: Vec::new(),
        })
    }

    pub fn run(&mut self) -> VResult<Value> {
        loop {
            let mut frame = self.stack.pop().ok_or(VError::EmptyStack)?;
            let vfunc =
                self.functions
                    .get(frame.function.inner_id())
                    .ok_or(VError::FunctionNotFound {
                        id: frame.function.inner_id() as u32,
                    })?;

            match frame.step_until_vm_is_needed(vfunc)? {
                function::FrameOut::FunctionCallPls { fid, args, dst } => {
                    self.memory.push(dst); // Where we need to later save the returned value
                    self.stack.push(frame);
                    let mut newfn = generate_frame(fid, &self.functions)?;
                    newfn.load_inputs(args);
                    self.stack.push(newfn);
                }
                function::FrameOut::Return(register) => {
                    let value = *frame.read_register(&register)?;
                    let dst = match self.memory.pop() {
                        Some(r) => r,
                        None => {
                            return Ok(value);
                        }
                    };
                    let lf = self.stack.last_mut().ok_or(VError::EmptyStack)?;
                    lf.store_value(value, &dst)?;
                }
            }
        }
    }
}


