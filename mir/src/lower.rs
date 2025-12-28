//! Lower HIR to MIR

use calamars_core::{
    Identifier,
    ids::{self, ExpressionId, SymbolId, TypeId},
};
use front::sematic::hir::{self, BinOp, Const, ItemId, SymbolKind};

use crate::{
    BBlock, BinaryOperator, BlockId, Function, FunctionId, VInstruct, VInstructionKind, ValueId,
    errors::MirErrors,
};

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

pub type MirRes<A> = Result<A, MirErrors>;

pub struct FunctionBuilder<'a> {
    ctx: &'a hir::Module,

    current_block: BlockId,

    blocks: Vec<BBlock>,
    instructions: Vec<VInstruct>,

    /// What valueid corresponds to some identifier
    locals: hashbrown::HashMap<SymbolId, ValueId>,
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(ctx: &'a hir::Module) -> Self {
        Self {
            ctx,
            current_block: BlockId(0),
            blocks: vec![],
            instructions: vec![],
            locals: hashbrown::HashMap::new(),
        }
    }

    pub fn block_mut(&mut self) -> MirRes<&mut BBlock> {
        let blockid = self.current_block.inner_id();
        self.blocks
            .get_mut(blockid)
            .ok_or(MirErrors::NoWorkingBlock)
    }

    fn new_block(&mut self) -> BlockId {
        self.blocks.push(BBlock::default());
        BlockId(self.blocks.len() - 1)
    }

    fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    fn push_instuction_get_index(&mut self, inst: VInstruct) -> ValueId {
        self.instructions.push(inst);
        ValueId(self.instructions.len() - 1)
    }

    fn terminate(&mut self, term: crate::Terminator) -> MirRes<()> {
        self.block_mut()?.with_term(term);
        Ok(())
    }

    pub fn terminate_br(&mut self, target: BlockId) -> MirRes<()> {
        let term = crate::Terminator::Br { target };
        self.terminate(term)
    }

    pub fn term_br_if(
        &mut self,
        condition: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    ) -> MirRes<()> {
        let term = crate::Terminator::BrIf {
            condition,
            then_target,
            else_target,
        };
        self.terminate(term)
    }

    pub fn term_ret(&mut self, value: ValueId) -> MirRes<()> {
        let term = crate::Terminator::Return(Some(value));
        self.terminate(term)
    }

    pub fn emit(&mut self, inst_kind: VInstructionKind) -> MirRes<ValueId> {
        let vid = self.push_instuction_get_index(VInstruct { kind: inst_kind });
        self.block_mut()?.with_instruct(vid);
        Ok(vid)
    }

    fn emit_unit(&mut self) -> MirRes<ValueId> {
        self.emit(VInstructionKind::Constant(crate::Consts::Unit))
    }

    fn emit_phi(
        &mut self,
        ty: ids::TypeId,
        incoming: Box<[(BlockId, ValueId)]>,
    ) -> MirRes<ValueId> {
        let kind = VInstructionKind::Phi { ty, incoming };
        self.emit(kind)
    }

    fn emit_literal(&mut self, constant: &Const) -> MirRes<ValueId> {
        let kind = VInstructionKind::Constant(match constant {
            hir::Const::I64(i) => crate::Consts::I64(*i),
            hir::Const::Bool(b) => crate::Consts::Bool(*b),
            hir::Const::String(sid) => crate::Consts::String(*sid),
        });
        self.emit(kind)
    }

    fn emit_binary(
        &mut self,
        operator: &BinOp,
        lhs: &ExpressionId,
        rhs: &ExpressionId,
    ) -> MirRes<ValueId> {
        let lhs = self.lower_expression_from_id(lhs)?;
        let rhs = self.lower_expression_from_id(rhs)?;
        let op = operator_map(operator);
        let kind = VInstructionKind::Binary { op, lhs, rhs };
        self.emit(kind)
    }

    fn emit_if(
        &mut self,
        expr_ty: ids::TypeId,
        predicate: &ExpressionId,
        then: &ExpressionId,
        otherwise: &ExpressionId,
    ) -> MirRes<ValueId> {
        let pred = self.lower_expression_from_id(&predicate)?;

        // Generate blocks for the if and for the then
        let ifb = self.new_block();
        let elb = self.new_block();
        let joinb = self.new_block();

        self.term_br_if(pred, ifb, elb)?;

        self.switch_to_block(ifb);
        let then_val = self.lower_expression_from_id(&then)?;
        self.terminate_br(joinb)?;

        self.switch_to_block(elb);
        let else_val = self.lower_expression_from_id(&otherwise)?;
        self.terminate_br(joinb)?;

        self.switch_to_block(joinb);
        self.emit_phi(expr_ty, Box::new([(ifb, then_val), (elb, else_val)]))
    }

    fn lower_expression_from_id(&mut self, expressionid: &ExpressionId) -> MirRes<ValueId> {
        let expression = self
            .ctx
            .exprs
            .get(*expressionid)
            .ok_or(MirErrors::ExpressionNotFound)?;

        let ty = self
            .ctx
            .expression_types
            .get(expressionid)
            .copied()
            .ok_or(MirErrors::CouldNotGetExpressionType)?;

        self.lower_expression(ty, expression)
    }

    /// Given some expression, turn it into a series of instructions, and return the ValueId where
    /// the result of the expression is.
    fn lower_expression(&mut self, ty: ids::TypeId, expression: &hir::Expr) -> MirRes<ValueId> {
        match expression {
            /// When we load an identifier, dont generate new instructions
            hir::Expr::Identifier { id, .. } => {
                self.locals.get(id).copied().ok_or(MirErrors::IdentNotFound)
            }
            hir::Expr::Literal { constant, .. } => self.emit_literal(constant),
            hir::Expr::BinaryOperation {
                operator, lhs, rhs, ..
            } => self.emit_binary(operator, lhs, rhs),
            hir::Expr::If {
                predicate,
                then,
                otherwise,
                ..
            } => self.emit_if(ty, predicate, then, otherwise),
            _ => todo!("Cannot yet handle: {:?}", expression),
        }
    }

    pub fn clear_all(&mut self) {
        self.locals.clear();
        self.blocks.clear();
        self.instructions.clear();
    }

    pub fn lower(
        &mut self,
        name: ids::IdentId,
        return_ty: ids::TypeId,
        params: &[SymbolId],
        body: ExpressionId,
    ) -> MirRes<Function> {
        self.clear_all();
        self.current_block = self.new_block();

        let mut params_inst = Vec::with_capacity(params.len());
        for (index, param) in params.iter().enumerate() {
            let sym = self
                .ctx
                .symbols
                .get(*param)
                .ok_or(MirErrors::ParamNotFound)?;

            let kind = VInstructionKind::Parameter {
                index: index as u16,
                ty: sym.ty_id(),
            };
            let vid = self.push_instuction_get_index(VInstruct { kind });
            self.block_mut()?.with_instruct(vid);
            self.locals.insert(*param, vid);
            params_inst.push(vid);
        }

        let body_val = self.lower_expression_from_id(&body)?;
        self.term_ret(body_val)?;

        Ok(Function {
            name,
            return_ty,
            params: params_inst,
            instructions: std::mem::take(&mut self.instructions),
            blocks: std::mem::take(&mut self.blocks),
        })
    }
}
