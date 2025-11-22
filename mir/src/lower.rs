//! Lower HIR to MIR

use calamars_core::ids::{self, ExpressionId, SymbolId, TypeId};
use front::sematic::hir::{self, ItemId, SymbolKind};

use crate::{
    BBlock, BinaryOperator, BindingId, BlockId, Function, FunctionArena, FunctionId, VInstruct,
    VInstructionKind, ValueId, errors::MirErrors,
};

use super::{BlockArena, InstructionArena};

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

/// Handle the process of building a function from the HIR context
pub struct MirBuilder<'a> {
    pub ctx: &'a hir::Module,

    current_block_id: BlockId,
    working_blocks: Vec<BlockId>,

    blocks: BlockArena,
    functions: FunctionArena,
    instructions: InstructionArena,
    locals: hashbrown::HashMap<ids::SymbolId, ValueId>,
}

impl<'a> MirBuilder<'a> {
    pub fn new(ctx: &'a hir::Module) -> Self {
        let mut blocks = BlockArena::new_unchecked();
        let current_block_id = blocks.push(BBlock::default());
        Self {
            ctx,
            current_block_id,
            working_blocks: vec![],
            blocks,
            functions: FunctionArena::new_unchecked(),
            instructions: InstructionArena::new_unchecked(),
            locals: hashbrown::HashMap::new(),
        }
    }

    fn switch_to(&mut self, b: BlockId) {
        self.current_block_id = b;
    }

    fn block_mut(&mut self) -> &mut BBlock {
        self.blocks.get_unchecked_mut(self.current_block_id)
    }

    fn new_block(&mut self) -> BlockId {
        let id = self.blocks.push(BBlock::default());
        self.working_blocks.push(id);
        id
    }

    fn term_br(&mut self, go: BlockId) {
        let term = crate::Terminator::Br { target: go };
        self.block_mut().with_term(term);
    }

    fn term_cond_br(&mut self, condition: ValueId, then: BlockId, otherwise: BlockId) {
        let term = crate::Terminator::BrIf {
            condition,
            then_target: then,
            else_target: otherwise,
        };
        self.block_mut().with_term(term);
    }

    fn term_ret(&mut self, value: ValueId) {
        let term = crate::Terminator::Return(Some(value));
        self.block_mut().with_term(term);
    }

    fn emit(&mut self, kind: VInstructionKind) -> ValueId {
        let instruct = self.instructions.push(VInstruct { dst: None, kind });
        self.block_mut().with_instruct(instruct);
        instruct
    }

    /// Emit Unit type as a constant
    fn emit_unit(&mut self) -> ValueId {
        self.emit(VInstructionKind::Constant(crate::Consts::Unit))
    }

    fn emit_phi(&mut self, ty: ids::TypeId, incoming: Box<[(BlockId, ValueId)]>) -> ValueId {
        let kind = VInstructionKind::Phi { ty, incoming };
        self.emit(kind)
    }

    fn lower_item(&mut self, item: &ItemId) {
        match item {
            ItemId::Expr(expression_id) => {
                self.lower_expression(*expression_id);
            }
            ItemId::Symbol(symbol_id) => {
                let _ = self.lower_binding(*symbol_id);
            }
        }
    }

    fn lower_block(&mut self, items: &[ItemId], final_expr: &Option<ids::ExpressionId>) -> ValueId {
        for item in items {
            self.lower_item(item);
        }

        match final_expr {
            Some(expr_id) => self.lower_expression(*expr_id),
            None => self.emit_unit(),
        }
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
            hir::Expr::Identifier { .. } => unreachable!("This case was handled above"),

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
            hir::Expr::If {
                predicate,
                then,
                otherwise,
                ..
            } => {
                // Lower the predicate, and end the block in a CondBr terminator
                let pred = self.lower_expression(*predicate);
                let then_block = self.new_block();
                let othr_block = self.new_block();
                self.term_cond_br(pred, then_block, othr_block);

                // A block where we join the two branches into one
                let joining_block = self.new_block();

                self.switch_to(then_block);
                let then = self.lower_expression(*then);
                self.term_br(joining_block);

                self.switch_to(othr_block);
                let otherwise = self.lower_expression(*otherwise);
                self.term_br(joining_block);

                self.switch_to(joining_block);

                let ty = self.ctx.expression_types.get(&expression_id).unwrap();
                return self.emit_phi(
                    *ty,
                    Box::from([(then_block, then), (othr_block, otherwise)]),
                );
            }
            hir::Expr::Block {
                items, final_expr, ..
            } => {
                return self.lower_block(items, final_expr);
            }
            _ => todo!("Cannot yet handle: {:?}", expression),
        };

        self.emit(kind)
    }

    fn lower_function(
        &mut self,
        name: ids::IdentId,
        return_ty: TypeId,
        params: &Box<[SymbolId]>,
        body: ExpressionId,
    ) -> Result<FunctionId, MirErrors> {
        self.locals.clear();
        self.working_blocks.clear();

        let entry = self.new_block();
        self.current_block_id = entry;

        for (i, param_id) in params.iter().enumerate() {
            let vk = VInstructionKind::Parameter { index: i as u8 };
            let vi = self.emit(vk);
            self.locals.insert(*param_id, vi);
        }

        let expr = self.lower_expression(body);
        self.term_ret(expr);

        let f = Function {
            name,
            return_ty,
            entry,
            blocks: self.working_blocks.clone(),
        };

        Ok(self.functions.push(f))
    }

    fn lower_binding(&mut self, binding_id: ids::SymbolId) -> Result<BindingId, MirErrors> {
        let s = self.ctx.symbols.get_unchecked(binding_id);
        let t = self.ctx.types.get_unchecked(s.ty_id());
        match &s.kind {
            SymbolKind::Variable { body, .. } => {
                let assignment = self.lower_expression(*body);
                self.locals.insert(binding_id, assignment);
                Ok(assignment.into())
            }
            SymbolKind::Function { params, body } => self
                .lower_function(s.ident_id(), t.function_output(), params, *body)
                .map(Into::into),
            SymbolKind::VariableUndeclared { .. } | SymbolKind::FunctionUndeclared { .. } => {
                Err(MirErrors::LoweringErr {
                    msg: "Cannot do mir lowering with undeclared var/fn".into(),
                })
            }
            _ => todo!(),
        }
    }

    pub fn lower_module(&mut self) -> Result<(), Vec<MirErrors>> {
        let mut errors = vec![];
        for binding in &self.ctx.roots {
            match self.lower_binding(*binding) {
                Ok(_) => {}
                Err(mir_err) => {
                    errors.push(mir_err);
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn blocks(&self) -> &BlockArena {
        &self.blocks
    }

    pub fn functions(&self) -> &FunctionArena {
        &self.functions
    }

    pub fn instructions(&self) -> &InstructionArena {
        &self.instructions
    }
}

#[cfg(test)]
mod test_lower {

    use indoc::indoc;

    use calamars_core::ids;
    use front::{
        sematic::hir::{
            self, ConstantStringArena, ExpressionArena, IdentArena, SymbolArena, TypeArena,
        },
        syntax::span::Span,
    };

    use crate::{VInstructionKind, lower::MirBuilder, printer::MirPrinter};

    fn make_context() -> hir::Module {
        hir::Module {
            id: ids::FileId::from(0),
            name: "TestingFile".to_owned(),
            roots: Box::new([]),
            types: TypeArena::new_checked(),
            const_str: ConstantStringArena::new_unchecked(),
            idents: IdentArena::new_unchecked(),
            symbols: SymbolArena::new_unchecked(),
            exprs: ExpressionArena::new_checked(),
            expression_types: hashbrown::HashMap::new(),
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

    fn if_stm(
        predicate: ids::ExpressionId,
        then: ids::ExpressionId,
        otherwise: ids::ExpressionId,
    ) -> hir::Expr {
        hir::Expr::If {
            predicate,
            then,
            otherwise,
            span: Span::dummy(),
            pred_span: Span::dummy(),
            then_span: Span::dummy(),
            othewise_span: Span::dummy(),
        }
    }

    #[test]
    fn lower_literal() {
        let mut context = make_context();

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
    fn lower_literal_txt() {
        let mut context = make_context();

        // 1
        let one = lit_i64(1);
        let one_id = context.exprs.push(one);

        let mut builder = MirBuilder::new(&context);
        let value_id = builder.lower_expression(one_id);
        builder.term_ret(value_id);

        let printer = MirPrinter::new(&builder.blocks, &builder.instructions, &builder.functions);
        let b = printer.fmt_block(&builder.current_block_id);
        let exp = indoc! {"
            bb0:
              %v0 = const 1
              return %v0
        "};
        assert_eq!(b, exp);
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

    #[test]
    fn simple_if() {
        /*
         * Try to lower the following:
         * if true then 1+1 else 4
         *
         * This should generate 4 blocks:
         * 1. define constant "true" and do conditional break
         * 2. define constant 1 and return 1+1
         * 3. define constant 4 and return it
         * 4. Phi instruction that assigns either 1 or 4
         * */

        let mut context = make_context();
        // Boolean Type
        let bt = context.types.intern(&hir::Type::Boolean);
        // Integer Type
        let it = context.types.intern(&hir::Type::Integer);

        // Make literals
        let t = lit_bool(true);
        let f = lit_i64(4);
        let o = lit_i64(1);
        let op = lit_i64(1);

        // Generate expression ids
        let t = context.exprs.push(t);
        let f = context.exprs.push(f);
        let o = context.exprs.push(o);
        let op = context.exprs.push(op);

        // Map expression ids to their types
        context.expression_types.insert(t, bt);
        context.expression_types.insert(f, it);
        context.expression_types.insert(o, it);
        context.expression_types.insert(op, it);

        // Binary expression
        let bin = bin_expr(hir::BinOp::Add, o, op);
        let bin = context.exprs.push(bin);
        context.expression_types.insert(bin, it);

        // If expression
        let cond = if_stm(t, bin, f);
        let cond = context.exprs.push(cond);
        context.expression_types.insert(cond, it);

        let mut builder = MirBuilder::new(&context);
        let if_lower = builder.lower_expression(cond);
        assert_eq!(builder.blocks.len(), 4);
    }
}
