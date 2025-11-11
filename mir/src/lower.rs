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
    pub fn new(ctx: &'a Context) -> Self {
        let mut s = Self {
            ctx,
            current_block_id: BlockId(0),
            blocks: BlockArena::new_unchecked(),
            instructions: InstructionArena::new_unchecked(),
            locals: hashbrown::HashMap::new(),
        };
        s.next_block();
        s
    }

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

#[cfg(test)]
mod test_lower {

    use calamars_core::ids;
    use front::{
        sematic::hir::{
            self, ConstantStringArena, ExpressionArena, IdentArena, SymbolArena, TypeArena,
        },
        syntax::span::Span,
    };

    use crate::{
        VInstructionKind,
        lower::{Context, MirBuilder},
    };

    fn make_context() -> Context {
        Context {
            types: TypeArena::new_checked(),
            const_str: ConstantStringArena::new_unchecked(),
            idents: IdentArena::new_unchecked(),
            symbols: SymbolArena::new_unchecked(),
            exprs: ExpressionArena::new_checked(),
        }
    }

    fn lit_i64(v: i64) -> hir::Expr {
        hir::Expr::Literal {
            constant: hir::Const::I64(v),
            span: Span::dummy(),
        }
    }

    fn lit_bool(v: bool) -> hir::Expr {
        hir::Expr::Literal {
            constant: hir::Const::Bool(v),
            span: Span::dummy(),
        }
    }

    fn ident(sym: ids::SymbolId) -> hir::Expr {
        hir::Expr::Identifier {
            id: sym,
            span: Span::dummy(),
        }
    }

    fn bin_expr(op: hir::BinOp, l: ids::ExpressionId, r: ids::ExpressionId) -> hir::Expr {
        hir::Expr::BinaryOperation {
            operator: op,
            lhs: l,
            rhs: r,
            span: Span::dummy(),
        }
    }

    #[test]
    fn lower_literal() {
        let mut context = make_context();

        // 1 + 3
        let one = lit_i64(1);
        let one_id = context.exprs.push(one);

        let mut builder = MirBuilder::new(&context);
        let value_id = builder.lower_expression(one_id);

        let block = builder.blocks.get_unchecked(builder.current_block_id);

        assert!(block.instructs.len() == 1);
        let id = block.instructs[0];
        let instruct = builder.instructions.get_unchecked(id);
        assert!(matches!(instruct.kind, VInstructionKind::Constant { .. }));
    }

    #[test]
    fn lower_simple_binary_expr() {
        let mut context = make_context();

        // 1 + 3
        let one = lit_i64(1);
        let one_id = context.exprs.push(one);
        let three = lit_i64(3);
        let three_id = context.exprs.push(three);

        let sum = bin_expr(hir::BinOp::Add, one_id, three_id);
        let sum_id = context.exprs.push(sum);

        let mut builder = MirBuilder::new(&context);
        let value_id = builder.lower_expression(sum_id);

        let block = builder.blocks.get_unchecked(builder.current_block_id);

        assert!(block.instructs.len() == 3);

        let zi = builder.instructions.get_unchecked(block.instructs[0]);
        let oi = builder.instructions.get_unchecked(block.instructs[1]);
        let ti = builder.instructions.get_unchecked(block.instructs[2]);

        // The desired block should be:
        //
        // v0 = const 1
        // v1 = const 2
        // v2 = + v0 v1
        assert!(matches!(zi.kind, VInstructionKind::Constant { .. }));
        assert!(matches!(oi.kind, VInstructionKind::Constant { .. }));
        assert!(matches!(ti.kind, VInstructionKind::Binary { .. }));
    }
}
