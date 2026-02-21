//! Lower HIR to MIR

use calamars_core::{
    Identifier, UncheckedArena,
    ids::{self, ExpressionId, SymbolId},
};
use front::sematic::hir::{self, BinOp, Const, ItemId, SymbolDec, SymbolKind};

use crate::{
    BBlock, BinaryOperator, BlockId, Callee, Function, FunctionId, Module, VInstruct,
    VInstructionKind, ValueId, errors::MirErrors,
};

fn operator_map(op: &hir::BinOp) -> BinaryOperator {
    match op {
        hir::BinOp::Add => BinaryOperator::Add,
        hir::BinOp::Sub => BinaryOperator::Sub,
        hir::BinOp::Mult => BinaryOperator::Times,
        hir::BinOp::Div => BinaryOperator::Div,
        hir::BinOp::Mod => BinaryOperator::Modulo,
        hir::BinOp::EqEq => BinaryOperator::EqEq,
        hir::BinOp::NotEqual => BinaryOperator::NotEqual,
        hir::BinOp::Greater => BinaryOperator::Greater,
        hir::BinOp::Geq => BinaryOperator::Geq,
        hir::BinOp::Less => BinaryOperator::Lesser,
        hir::BinOp::Leq => BinaryOperator::Leq,
        hir::BinOp::And => BinaryOperator::And,
        hir::BinOp::Or => BinaryOperator::Or,
        hir::BinOp::Xor => BinaryOperator::Xor,
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

    function_map: &'a hashbrown::HashMap<ids::IdentId, FunctionId>,
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(
        ctx: &'a hir::Module,
        function_map: &'a hashbrown::HashMap<ids::IdentId, FunctionId>,
    ) -> Self {
        Self {
            ctx,
            current_block: BlockId(0),
            blocks: vec![],
            instructions: vec![],
            locals: hashbrown::HashMap::new(),
            function_map,
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

    pub fn emit(&mut self, inst_kind: VInstructionKind) -> MirRes<(ValueId, BlockId)> {
        let vid = self.push_instuction_get_index(VInstruct { kind: inst_kind });
        self.block_mut()?.with_instruct(vid);
        Ok((vid, self.current_block))
    }

    fn emit_unit(&mut self) -> MirRes<(ValueId, BlockId)> {
        self.emit(VInstructionKind::Constant(crate::Consts::Unit))
    }

    fn emit_phi(
        &mut self,
        ty: ids::TypeId,
        incoming: Box<[(BlockId, ValueId)]>,
    ) -> MirRes<(ValueId, BlockId)> {
        let kind = VInstructionKind::Phi { ty, incoming };
        self.emit(kind)
    }

    fn emit_literal(&mut self, constant: &Const) -> MirRes<(ValueId, BlockId)> {
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
    ) -> MirRes<(ValueId, BlockId)> {
        let (lhs, _) = self.lower_expression_from_id(lhs)?;
        let (rhs, _) = self.lower_expression_from_id(rhs)?;
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
    ) -> MirRes<(ValueId, BlockId)> {
        let (pred, _) = self.lower_expression_from_id(&predicate)?;

        // Generate blocks for the if and for the then
        let ifb = self.new_block();
        let elb = self.new_block();
        let joinb = self.new_block();

        self.term_br_if(pred, ifb, elb)?;

        self.switch_to_block(ifb);
        let (then_vid, then_block) = self.lower_expression_from_id(&then)?;
        self.terminate_br(joinb)?;

        self.switch_to_block(elb);
        let (otherwise_vid, otherwise_block) = self.lower_expression_from_id(&otherwise)?;
        self.terminate_br(joinb)?;

        self.switch_to_block(joinb);
        let pairs = [(then_block, then_vid), (otherwise_block, otherwise_vid)];
        self.emit_phi(expr_ty, Box::new(pairs))
    }

    fn emit_block(
        &mut self,
        _expr_ty: ids::TypeId,
        items: &Box<[ItemId]>,
        expr: &Option<ids::ExpressionId>,
    ) -> MirRes<(ValueId, BlockId)> {
        for i in items {
            match i {
                ItemId::Expr(expression_id) => {
                    self.lower_expression_from_id(expression_id)?;
                }
                ItemId::Symbol(symbol_id) => {
                    let x = self.ctx.symbols.get_unchecked(*symbol_id).kind.clone();
                    if let SymbolKind::Defn {
                        declaration: SymbolDec { inputs, body },
                        ..
                    } = x
                        && inputs.is_empty()
                    {
                        let (v, _) = self.lower_expression_from_id(&body)?;
                        self.locals.insert(*symbol_id, v);
                    } else {
                        panic!("Function declarations are not allowed in blocks yet, sorry :(");
                    }
                }
            }
        }

        // Return the position of the last block
        if let Some(eid) = *expr {
            self.lower_expression_from_id(&eid)
        } else {
            self.emit_unit()
        }
    }

    fn lower_expression_from_id(
        &mut self,
        expressionid: &ExpressionId,
    ) -> MirRes<(ValueId, BlockId)> {
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

    /// When lowering to MIR we will no longer require that functions be
    /// of exactly one input and one output. That is, given the following fn:
    ///
    /// ```txt
    /// def add (x : Int) (y : Int) : Int x + y
    /// ```
    ///
    /// When we call `add 2 3`, which in the HIR is interpreted as `(add 2) 3`
    /// we will lower it to a single function call, not 2 of them.
    ///
    /// For now, we have a few restrictions:
    /// 1. We will never save intermediary values (i.e. z = add 2 is invalid)
    ///
    /// The restrictions will grow and shrink, but ultimately, we need them all
    /// gone relatively soon.
    ///
    /// Returns: (function we are calling, inputs in order)
    /// add 2 3 |-> (add, [2, 3])
    fn flatten_function_calls(
        &self,
        f: &ExpressionId,
        input: &ExpressionId,
    ) -> (SymbolId, Vec<ExpressionId>) {
        match self.ctx.exprs.get_unchecked(*f) {
            hir::Expr::Call {
                f,
                input: nested_input,
                ..
            } => {
                let (callee, mut inputs) = self.flatten_function_calls(f, nested_input);
                inputs.push(*input);
                (callee, inputs)
            }
            hir::Expr::Identifier { id, .. } => (*id, vec![*input]),
            _ => todo!(),
        }
    }

    fn emit_function_call(
        &mut self,
        f: &ExpressionId,
        input: &ExpressionId,
    ) -> MirRes<(ValueId, BlockId)> {
        let (fn_symbol_id, inputs) = self.flatten_function_calls(f, input);
        let fn_symbol = self.ctx.symbols.get_unchecked(fn_symbol_id);
        let fn_ident = fn_symbol.name;
        let fn_id = self.function_map.get(&fn_ident).expect("Did not lower fn");
        let callee = crate::Callee::Function(*fn_id);

        let mut args = Vec::with_capacity(inputs.len());
        for i in inputs.iter() {
            let (v, _) = self.lower_expression_from_id(i)?;
            args.push(v);
        }

        let return_ty = *self
            .ctx
            .expression_types
            .get(f)
            .expect("function expression should have had a type ...");

        self.emit(VInstructionKind::Call {
            callee,
            args,
            return_ty,
        })
    }

    /// Given some expression, turn it into a series of instructions, and return the ValueId where
    /// the result of the expression is.
    fn lower_expression(
        &mut self,
        ty: ids::TypeId,
        expression: &hir::Expr,
    ) -> MirRes<(ValueId, BlockId)> {
        match expression {
            // When we load an identifier, don't generate new instructions
            hir::Expr::Identifier { id, .. } => {
                let vid = self
                    .locals
                    .get(id)
                    .copied()
                    .ok_or(MirErrors::IdentNotFound)?;
                Ok((vid, self.current_block))
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
            hir::Expr::Block {
                items, final_expr, ..
            } => self.emit_block(ty, items, final_expr),
            hir::Expr::Call { f, input, .. } => self.emit_function_call(f, input),
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
        id: FunctionId,
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
                ty: sym.ty,
            };
            let vid = self.push_instuction_get_index(VInstruct { kind });
            self.block_mut()?.with_instruct(vid);
            self.locals.insert(*param, vid);
            params_inst.push(vid);
        }

        let (body_val, _) = self.lower_expression_from_id(&body)?;
        self.term_ret(body_val)?;

        Ok(Function {
            name,
            return_ty,
            params: params_inst,
            instructions: std::mem::take(&mut self.instructions),
            blocks: std::mem::take(&mut self.blocks),
            id,
        })
    }
}

pub struct ModuleBuilder<'a> {
    ctx: &'a hir::Module,
    functions: UncheckedArena<Function, FunctionId>,

    // internal values for building
    function_map: hashbrown::HashMap<ids::IdentId, FunctionId>,
}

impl<'a> ModuleBuilder<'a> {
    pub fn new(ctx: &'a hir::Module) -> Self {
        Self {
            ctx,
            functions: UncheckedArena::new_unchecked(),
            function_map: hashbrown::HashMap::new(),
        }
    }

    /// Generate the functionids before we starts lowering, so that you don't need
    /// to define before use.
    pub fn handle_headers(&mut self) {
        for symbol_id in &self.ctx.roots {
            let symbol = self.ctx.symbols.get_unchecked(*symbol_id);
            if let SymbolKind::Defn {
                declaration: SymbolDec { .. },
                ..
            } = &symbol.kind {
                let id = FunctionId::from(self.function_map.len());
                self.function_map.insert(symbol.name, id);
            }
        }
    }

    pub fn lower_entire_module(&mut self) -> MirRes<()> {
        self.handle_headers();

        for symbol_id in &self.ctx.roots {
            let symbol = self.ctx.symbols.get_unchecked(*symbol_id);
            let name = symbol.name;
            let return_ty = symbol.ty;

            let (params, body) = match &symbol.kind {
                SymbolKind::Defn { declaration, .. } => (&declaration.inputs, declaration.body),
                _ => continue,
            };

            let id = *self.function_map.get(&name).expect("Function id missing");
            let mut builder = FunctionBuilder::new(self.ctx, &self.function_map);
            let func = builder.lower(name, return_ty, params, body, id)?;
            self.functions.push(func);
        }
        Ok(())
    }

    /// Lower a function and return its id in the arena
    pub fn lower_function(
        &mut self,
        name: ids::IdentId,
        return_ty: ids::TypeId,
        params: &[SymbolId],
        body: ExpressionId,
    ) -> MirRes<FunctionId> {
        let id = FunctionId::from(self.functions.len());
        let mut builder = FunctionBuilder::new(&self.ctx, &self.function_map);
        let lower = builder.lower(name, return_ty, params, body, id)?;
        self.functions.push(lower);
        Ok(id)
    }

    /// Generate a final module
    pub fn finish(self) -> Module {
        Module::new(self.functions)
    }
}
