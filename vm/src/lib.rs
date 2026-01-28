//! A virtual machine to run Calamars
//!
//! For now, this file is meant to "just sort of work", and thus it is VERY inneficient. Once the
//! project reaches a state where the front end is semi-stable, and a stdlib exists, much
//! optimization needs to be done here.

use crate::function::{Frame, VFunction};

mod bytecode;
mod errors;
mod function;

// A register
#[derive(Clone, Debug)]
pub struct Register(u8);

impl From<u8> for Register {
    fn from(value: u8) -> Self {
        Register(value)
    }
}

impl Register {
    pub fn inner_id(&self) -> u8 {
        self.0
    }
}

/// Stack allocated values
#[derive(Clone, Debug)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Char(char),
}

/// Virtual Machine
pub struct VMachine {
    functions: Box<[VFunction]>,
    stack: Vec<Frame>,
}
