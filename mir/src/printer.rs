use std::fmt;

use calamars_core::Identifier;

use crate::{
    BinaryOperator, BitwiseBinaryOperator, BlockArena, BlockId, Callee, Consts, Function,
    InstructionArena, Terminator, UnaryOperator, VInstructionKind, ValueId,
};

pub struct MirPrinter<'a> {
    pub blocks: &'a super::BlockArena,
    pub insts: &'a super::InstructionArena,
}

impl<'a> MirPrinter<'a> {
    pub fn new(blocks: &'a BlockArena, insts: &'a InstructionArena) -> Self {
        Self { blocks, insts }
    }

    #[inline]
    fn v(&self, id: ValueId) -> String {
        format!("v{}", id.0)
    }

    #[inline]
    fn bb(&self, id: BlockId) -> String {
        format!("bb{}", id.0)
    }

    /// Format a Value Producing instruction
    pub fn fmt_vinst(&self, kind: &VInstructionKind) -> String {
        match kind {
            VInstructionKind::Constant(c) => match c {
                Consts::I64(i) => format!("const {}", i),
                Consts::Bool(b) => format!("const {}", b),
                // TODO: Shuold we resolve for the actual text here ?
                Consts::String(s) => format!("const str#{:?}", s),
                Consts::Unit => "const ()".to_string(),
            },
            VInstructionKind::ConstDataPointer { data } => {
                format!("data.ptr @data{}", data.0)
            }
            VInstructionKind::Unary { op, on } => {
                let op_s = match op {
                    UnaryOperator::Not => "not",
                    UnaryOperator::Negate => "neg",
                };
                format!("{op_s} {}", self.v(*on))
            }
            VInstructionKind::Binary { op, lhs, rhs } => {
                let op_s = match op {
                    BinaryOperator::Add => "+",
                    BinaryOperator::Sub => "-",
                    BinaryOperator::Times => "*",
                    BinaryOperator::Div => "/",
                    BinaryOperator::Modulo => "%",
                    BinaryOperator::EqEq => "==",
                    BinaryOperator::NotEqual => "!=",
                    BinaryOperator::Greater => ">",
                    BinaryOperator::Geq => ">=",
                    BinaryOperator::Lesser => "<",
                    BinaryOperator::Leq => "<=",
                };
                format!("{} {} {}", self.v(*lhs), op_s, self.v(*rhs))
            }
            VInstructionKind::BitwiseBinary { op, lhs, rhs } => {
                let op_s = match op {
                    BitwiseBinaryOperator::And => "and",
                    BitwiseBinaryOperator::Xor => "xor",
                    BitwiseBinaryOperator::Or => "or",
                };
                format!("{} {op_s} {}", self.v(*lhs), self.v(*rhs))
            }
            VInstructionKind::Call {
                callee,
                args,
                return_ty,
            } => {
                let callee_s = match callee {
                    Callee::Function(fid) => format!("fn#{}", fid.inner_id()),
                    Callee::Extern(name) => format!("@{}", name),
                };
                let args_s = args
                    .iter()
                    .map(|a| self.v(*a))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("call {callee_s}({args_s}) : ty#{}", return_ty.inner_id())
            }
            VInstructionKind::Phi { ty, incoming } => {
                let cs = incoming
                    .iter()
                    .map(|(b, v)| format!("{}: %v{}", self.bb(*b), self.v(*v)))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("phi ty#{} [{}]", ty.inner_id(), cs)
            }
        }
    }

    /// Format at terminator
    pub fn fmt_term(&self, t: &Terminator) -> String {
        match t {
            Terminator::Return(Some(v)) => format!("return {}", self.v(*v)),
            Terminator::Return(None) => "return".to_string(),
            Terminator::Br { target } => {
                format!("br {}", self.bb(*target))
            }
            Terminator::BrIf {
                condition,
                then_target,
                else_target,
            } => {
                format!(
                    "br_if {}, then: {} else: {}",
                    self.v(*condition),
                    self.bb(*then_target),
                    self.bb(*else_target),
                )
            }
        }
    }
}
