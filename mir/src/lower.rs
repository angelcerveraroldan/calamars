//! Lower HIR to MIR

use calamars_core::ids;
use front::sematic::hir;

use crate::{BBlock, BinaryOperator, BlockId, Function, VInstruct, VInstructionKind, ValueId};

type InstructionArena = calamars_core::UncheckedArena<VInstruct, ValueId>;
type BlockArena = calamars_core::UncheckedArena<BBlock, BlockId>;

fn operator_map(op: &hir::BinOp) -> BinaryOperator {
    match op {
        hir::BinOp::Add => BinaryOperator::Add,
        hir::BinOp::Sub => BinaryOperator::Sub,
        hir::BinOp::Mult => BinaryOperator::Times,
        hir::BinOp::Div => BinaryOperator::Div,
        hir::BinOp::Mod => BinaryOperator::Modulo,
        hir::BinOp::EqEq => BinaryOperator::EqEq,
    }
}

/// Immutable data outputted by the HIR
pub struct Context {
    pub types: hir::TypeArena,
    pub const_str: hir::ConstantStringArena,
    pub idents: hir::IdentArena,
    pub symbols: hir::SymbolArena,
    pub exprs: hir::ExpressionArena,
}

/// Handle the process of building a function from the HIR context
pub struct MirBuilder<'a> {
    pub ctx: &'a Context,
    current_block_id: BlockId,

    blocks: BlockArena,
    instructions: InstructionArena,
    locals: hashbrown::HashMap<ids::SymbolId, ValueId>,
}

impl<'a> MirBuilder<'a> {
    /// Start working on the next block
    fn next_block(&mut self) {
        let default_block = BBlock {
            params: vec![],
            instructs: vec![],
            finally: crate::Terminator::Return(None),
        };
        let id = self.blocks.push(default_block);
        self.current_block_id = id;
    }

    /// Finalize the current block, and insert it into a function
    fn insert_block_to_fn(&mut self, func: &mut Function) {
        func.blocks.push(self.current_block_id);
        self.next_block();
    }

    /// Add a return type to the current block
    fn finish_block_with_return(&mut self, vid: ValueId) {
        self.blocks.get_unchecked_mut(self.current_block_id).finally =
            crate::Terminator::Return(Some(vid))
    }

    /// Given some (value producing) expression from the HIR, turn it into a `VInstruct`, add it to
    /// the arena and return its id.
    ///
    /// In the case that the expression was just referencing an ident, just return the valueid
    /// associated with said ident.
    fn lower_expression(&mut self, expression_id: ids::ExpressionId) -> ValueId {
        let expression = self.ctx.exprs.get_unchecked(expression_id);
        if let hir::Expr::Identifier { id, .. } = expression {
            return *self.locals.get(id).unwrap();
        }

        let kind = match expression {
            // The case is handled differently, as it does not generate a new instruction, but
            // uses an old one.
            hir::Expr::Identifier { id, .. } => unreachable!("This case was handled above"),

            hir::Expr::Literal { constant, .. } => VInstructionKind::Constant(match constant {
                hir::Const::I64(i) => crate::Consts::I64(*i),
                hir::Const::Bool(b) => crate::Consts::Bool(*b),
                hir::Const::String(sid) => crate::Consts::String(*sid),
            }),
            hir::Expr::BinaryOperation {
                operator,
                lhs,
                rhs,
                span,
            } => {
                let lhs = self.lower_expression(*lhs);
                let rhs = self.lower_expression(*rhs);
                let op = operator_map(operator);
                VInstructionKind::Binary { op, lhs, rhs }
            }
            _ => todo!(),
        };

        let instruction = VInstruct {
            dst: None,
            kind,
            span: expression.get_span().unwrap(),
        };

        let value_id = self.instructions.push(instruction);
        self.blocks
            .get_unchecked_mut(self.current_block_id)
            .instructs
            .push(value_id);

        value_id
    }
}
