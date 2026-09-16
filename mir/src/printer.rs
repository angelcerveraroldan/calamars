use calamars_core::{
    Identifier,
    types::{TypeArena, type_id_stringify},
};
use front::semantic::hir::IdentArena;

use std::fmt::Write;

use crate::{
    BinaryOperator, BitwiseBinaryOperator, BlockId, BlockJump, Callee, Consts, Function,
    FunctionId, Terminator, UnaryOperator, VInstructionKind, ValueId,
};

pub struct MirPrinter<'a> {
    functions: &'a [Function],
    type_arena: &'a TypeArena,
    ident_arena: &'a IdentArena,
}

impl<'a> MirPrinter<'a> {
    pub fn new(
        functions: &'a [Function],
        type_arena: &'a TypeArena,
        ident_arena: &'a IdentArena,
    ) -> Self {
        Self {
            functions,
            type_arena,
            ident_arena,
        }
    }

    #[inline]
    fn v(&self, id: ValueId) -> String {
        format!("%v{}", id.0)
    }

    #[inline]
    fn bb(&self, function: &Function, id: BlockId) -> String {
        match id.inner_id() {
            0 => format!("_start"),
            id => format!("bb{}", id),
        }
    }

    fn bb_params(&self, function: &Function, id: BlockId) -> String {
        let mut s = String::new();
        let block = &function.blocks[id.inner_id()];
        let param_count = block.params.len();
        if param_count == 0 {
            return s;
        }
        let _ = write!(s, "(");
        for (index, param) in block.params.iter().enumerate() {
            let val = function.instructions.get(param.inner_id()).unwrap();
            let tystr = type_id_stringify(self.type_arena, val.vtype);
            let _ = write!(s, "{} :: {}", self.v(*param), tystr);
            if index != param_count - 1 {
                let _ = write!(s, ", ");
            }
        }
        let _ = write!(s, ")");
        s
    }

    pub fn fmt_call(&self, callee: &Callee, args: &Vec<ValueId>) -> String {
        let callee_s = match callee {
            Callee::Function(fid) => format!("fn#{}", fid.inner_id()),
            Callee::Extern(name) => format!("@{name}"),
        };
        let args_s = args
            .iter()
            .map(|a| self.v(*a))
            .collect::<Vec<_>>()
            .join(", ");
        format!("call {callee_s}({args_s})")
    }

    /// Format a Value Producing instruction
    pub fn fmt_vinst(&self, kind: &VInstructionKind) -> String {
        match kind {
            VInstructionKind::Constant(c) => match c {
                Consts::I64(i) => format!("const {i}"),
                Consts::Bool(b) => format!("const {b}"),
                // TODO: Shuold we resolve for the actual text here ?
                Consts::String(s) => format!("const str#{s:?}"),
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
                    BinaryOperator::And => "and",
                    BinaryOperator::Or => "or",
                    BinaryOperator::Xor => "xor",
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
            VInstructionKind::Call { callee, args } => self.fmt_call(callee, args),
            VInstructionKind::StructInit { ds_id, fields } => {
                let fields_s = fields
                    .iter()
                    .map(|field| self.v(*field))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("struct #{} {{{fields_s}}}", ds_id.inner_id())
            }
            VInstructionKind::Parameter { index, .. } => {
                format!("param #{index}")
            }
            VInstructionKind::ExtractField {
                source,
                ds_id,
                index,
            } => {
                format!("load {} #{}->#{index}", self.v(*source), ds_id.inner_id())
            }
        }
    }

    pub fn fmt_jump(&self, function: &Function, jump: &BlockJump) -> String {
        let argsfmt = jump
            .args
            .iter()
            .map(|vid| self.v(*vid))
            .collect::<Vec<_>>()
            .join(", ");

        format!("br {}({}):", self.bb(function, jump.target), argsfmt)
    }

    /// Format at terminator
    pub fn fmt_term(&self, function: &Function, t: &Terminator) -> String {
        match t {
            Terminator::Return(Some(v)) => format!("return {}", self.v(*v)),
            Terminator::Return(None) => "return".to_string(),
            Terminator::Call { callee, args } => format!("return {}", self.fmt_call(callee, args)),
            Terminator::Br { jump } => self.fmt_jump(function, jump),
            Terminator::BrIf {
                condition,
                then_jump,
                else_jump,
            } => {
                format!(
                    "br_if {}, then: {} else: {}",
                    self.v(*condition),
                    self.fmt_jump(function, then_jump),
                    self.fmt_jump(function, else_jump),
                )
            }
        }
    }

    pub fn fmt_block(&self, function: &Function, b: &BlockId) -> String {
        let mut s = String::new();
        let block = function.blocks.get(b.inner_id()).unwrap();
        let _ = writeln!(
            s,
            "{}{}:",
            self.bb(function, *b),
            self.bb_params(function, *b)
        );

        for inst in &block.instructs {
            let val = function.instructions.get(inst.inner_id()).unwrap();
            let rhs = self.fmt_vinst(&val.kind);
            let tystr = type_id_stringify(self.type_arena, val.vtype);
            let _ = writeln!(s, "  {} :: {} = {}", self.v(*inst), tystr, rhs);
        }
        if let Some(t) = &block.finally {
            let _ = writeln!(s, "  {}", self.fmt_term(function, t));
        }
        s
    }

    pub fn fmt_function_id(&self, fid: FunctionId) -> String {
        let f = self.functions.get(fid.inner_id()).unwrap();
        self.fmt_function(f)
    }

    pub fn fmt_function(&self, f: &Function) -> String {
        let mut s = String::new();
        let fname = self.ident_arena.get_unchecked(f.name);
        let input = f
            .dsign
            .params
            .iter()
            .map(|id| type_id_stringify(self.type_arena, *id))
            .collect::<Vec<_>>()
            .join(",");
        let output = type_id_stringify(self.type_arena, f.dsign.result);
        let _ = writeln!(s, "func @{} :: ({}) -> {} {{", fname, input, output);
        for bid in (0..f.blocks.len()).map(BlockId::from) {
            let blockfmt = self.fmt_block(f, &bid);
            s.push_str(&blockfmt);
        }

        let _ = writeln!(s, "}}");
        s
    }

    pub fn fmt_all_functions(&self) -> String {
        let mut out = String::new();
        for f in self.functions {
            out.push_str(&self.fmt_function(f));
            out.push('\n');
        }
        out
    }
}
