//! Lower AST to HIR

use calamars_core::ids;

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, Symbol, SymbolBuilder, SymbolDec, SymbolKind, take_inputs},
    },
    syntax::{
        ast::{self, Declaration},
        span::Span,
    },
};

fn lower_str_to_type(s: &str, span: &Span) -> Result<hir::Type, SemanticError> {
    Ok(match s {
        "Int" | "Integer" => hir::Type::Integer,
        "Float" | "Real" => hir::Type::Float,
        "Bool" | "Boolean" => hir::Type::Boolean,
        "String" => hir::Type::String,
        "Char" => hir::Type::Char,
        "Unit" | "()" => hir::Type::Unit,
        other => {
            return Err(SemanticError::TypeNotFound {
                type_name: other.to_string(),
                span: *span,
            });
        }
    })
}

pub struct HirBuilder {
    pub identifiers: hir::IdentArena,
    pub expressions: hir::ExpressionArena,
    pub symbols: hir::SymbolArena,

    /// Scopes are used for shadowing. When in the same scope we do not allow shadowing.
    scopes: Vec<hashbrown::HashMap<ids::IdentId, ids::SymbolId>>,

    /// Errors, if any are present, we cannot proceed with compilation
    pub diag_err: Vec<SemanticError>,
}

impl Default for HirBuilder {
    fn default() -> Self {
        let mut d = Self {
            identifiers: hir::IdentArena::new_unchecked(),
            expressions: hir::ExpressionArena::new_checked(),
            symbols: hir::SymbolArena::new_unchecked(),
            scopes: vec![],
            diag_err: vec![],
        };
        d.push_scope();
        d
    }
}

impl HirBuilder {
    pub fn errors(&self) -> &Vec<SemanticError> {
        &self.diag_err
    }

    fn insert_error(&mut self, err: SemanticError) {
        self.diag_err.push(err);
    }

    fn push_scope(&mut self) {
        self.scopes.push(hashbrown::HashMap::default());
    }

    /// Delete the current scope, unless this is the last scope, in which case nothing will
    /// happen
    fn pop_scope(&mut self) {
        if self.scopes.len() <= 1 {
            return;
        }
        self.scopes.pop();
    }

    fn lower_ident(&mut self, ident: &ast::Ident) -> ids::IdentId {
        let identifier_name = ident.ident().to_string();
        self.identifiers.intern(&identifier_name)
    }

    fn lower_expression(
        &mut self,
        expr: &ast::Expression,
        global_ctx: &mut hir::GlobalContext,
    ) -> ids::ExpressionId {
        let lowered = self.lower_expression_inner(expr, global_ctx);
        self.expressions.push(lowered)
    }

    fn lower_expression_inner(
        &mut self,
        expr: &ast::Expression,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        match expr {
            ast::Expression::Literal(literal) => self.lower_literal(literal, global_ctx),
            ast::Expression::Identifier(ident) => self.lower_identifier_expr(ident),
            ast::Expression::BinaryOp(binary_op) => self.lower_binary_expr(binary_op, global_ctx),
            ast::Expression::Apply(apply) => self.lower_apply_expr(apply, global_ctx),
            ast::Expression::IfStm(ifstm) => self.lower_if_expr(ifstm, global_ctx),
            ast::Expression::Block(block) => self.lower_block(block, global_ctx),
            ast::Expression::Error(_) => hir::Expr::Err,
            other => {
                self.insert_error(SemanticError::NotSupported {
                    msg: "Expression not yet supported",
                    span: other.span(),
                });
                hir::Expr::Err
            }
        }
    }

    fn lower_block(
        &mut self,
        block: &ast::CompoundExpression,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        self.push_scope();

        let mut items = vec![];
        let mut builders = hashbrown::HashMap::new();
        for item in block.items.iter() {
            match item {
                ast::Item::Declaration(decl) => match decl {
                    Declaration::TypeSignature {
                        docs: _,
                        name,
                        dtype,
                    } => {
                        let sb = self.lower_type_declaration(dtype, name, global_ctx);
                        let old = builders.insert(sb.ident_id(), sb);
                        if let Some(original) = old {
                            self.insert_error(SemanticError::Redeclaration {
                                original_span: original.span_name_type,
                                redec_span: name.span(),
                            })
                        }
                    }
                    Declaration::Binding { name, params, body } => {
                        let name_id = self.lower_ident(name);
                        let builder = builders.remove(&name_id);
                        let Some(builder) = builder else {
                            self.insert_error(SemanticError::TypeMissing {
                                for_identifier: name.span(),
                            });
                            continue;
                        };

                        match self
                            .lower_binding_declaration(name, params, body, builder, global_ctx)
                        {
                            Ok(symbol_id) => {
                                items.push(hir::ItemId::Symbol(symbol_id));
                            }
                            Err(err) => self.insert_error(err),
                        }
                    }
                },
                ast::Item::Expression(expr) => {
                    let expr_id = self.lower_expression(expr, global_ctx);
                    items.push(hir::ItemId::Expr(expr_id));
                }
            }
        }

        for (_, builder) in builders {
            self.insert_error(SemanticError::MissingDeclaration {
                name: builder.name,
                type_declaration: builder.span_name_type,
            });
        }

        let final_expr = block
            .final_expr
            .as_ref()
            .map(|expr| self.lower_expression(expr, global_ctx));

        self.pop_scope();
        hir::Expr::Block {
            items: items.into(),
            final_expr,
            span: block.total_span(),
        }
    }

    fn lower_literal(
        &mut self,
        literal: &ast::Literal,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        let constant = match literal.kind() {
            ast::LiteralKind::Integer(i) => hir::Const::I64(*i),
            ast::LiteralKind::Boolean(b) => hir::Const::Bool(*b),
            ast::LiteralKind::String(s) => hir::Const::String(global_ctx.const_str.intern(s)),
            _ => return hir::Expr::Err,
        };
        hir::Expr::Literal {
            constant,
            span: literal.span(),
        }
    }

    fn resolve(&self, ident_id: ids::IdentId, span: Span) -> Result<ids::SymbolId, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(&ident_id) {
                return Ok(*id);
            }
        }
        // We know that the identifier string must exist (we must have found it to call this
        // function)
        let name = self.identifiers.get_unchecked(ident_id).clone();
        Err(SemanticError::IdentNotFound { name, span })
    }

    fn insert_symbol_to_current_scope(
        &mut self,
        ident: ids::IdentId,
        symbol_id: ids::SymbolId,
        span: Span,
    ) -> Result<(), SemanticError> {
        // Not sure how to recover here
        let scope = self.scopes.last_mut().ok_or(SemanticError::InternalError {
            msg: "Lowering to HIR, scopes is empty",
            span,
        })?;
        // If the name was already in use, then we will log an error
        if let Some(original) = scope.insert(ident, symbol_id) {
            let original = self.symbols.get_unchecked(original);
            self.insert_error(SemanticError::Redeclaration {
                original_span: original.span(),
                redec_span: span,
            });
        }
        Ok(())
    }

    fn lower_identifier_expr(&mut self, ident: &ast::Ident) -> hir::Expr {
        let span = ident.span();
        let name = ident.ident();
        let ident_id = self.identifiers.intern(&name.to_string());
        match self.resolve(ident_id, span) {
            Ok(id) => hir::Expr::Identifier { id, span },
            Err(err) => {
                self.insert_error(err);
                hir::Expr::Err
            }
        }
    }

    fn lower_binary_expr(
        &mut self,
        binary_op: &ast::BinaryOp,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        let operator = match binary_op.operator() {
            ast::BinaryOperator::Add => hir::BinOp::Add,
            ast::BinaryOperator::Sub => hir::BinOp::Sub,
            ast::BinaryOperator::Times => hir::BinOp::Mult,
            ast::BinaryOperator::Div => hir::BinOp::Div,
            ast::BinaryOperator::EqEq => hir::BinOp::EqEq,
            ast::BinaryOperator::NotEqual => hir::BinOp::NotEqual,
            ast::BinaryOperator::Mod => hir::BinOp::Mod,
            ast::BinaryOperator::Greater => hir::BinOp::Greater,
            ast::BinaryOperator::Geq => hir::BinOp::Geq,
            ast::BinaryOperator::Less => hir::BinOp::Less,
            ast::BinaryOperator::Leq => hir::BinOp::Leq,
            ast::BinaryOperator::Or => hir::BinOp::Or,
            ast::BinaryOperator::Xor => hir::BinOp::Xor,
            ast::BinaryOperator::And => hir::BinOp::And,
            _ => return hir::Expr::Err,
        };
        let lhs = self.lower_expression(binary_op.lhs(), global_ctx);
        let rhs = self.lower_expression(binary_op.rhs(), global_ctx);
        hir::Expr::BinaryOperation {
            operator,
            lhs,
            rhs,
            span: binary_op.span(),
        }
    }

    fn lower_apply_expr(
        &mut self,
        apply: &ast::Apply,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        let f = self.lower_expression(apply.callable(), global_ctx);
        let input_expr = apply.input();
        let input = self.lower_expression(&input_expr, global_ctx);
        hir::Expr::Call {
            f,
            input,
            span: apply.span(),
        }
    }

    fn lower_if_expr(
        &mut self,
        ifstm: &ast::IfStm,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Expr {
        let predicate = self.lower_expression(ifstm.pred().as_ref(), global_ctx);
        let then = self.lower_expression(ifstm.then_expr().as_ref(), global_ctx);
        let otherwise = self.lower_expression(ifstm.else_expr().as_ref(), global_ctx);
        hir::Expr::If {
            predicate,
            then,
            otherwise,
            span: ifstm.span(),
            pred_span: ifstm.pred_span(),
            then_span: ifstm.then_span(),
            othewise_span: ifstm.else_span(),
        }
    }

    fn lower_type(
        &mut self,
        ast_type: &ast::Type,
        global_ctx: &mut hir::GlobalContext,
    ) -> ids::TypeId {
        let lowered = match ast_type {
            ast::Type::Error(_) => hir::Type::Error,
            ast::Type::Unit(_) => hir::Type::Unit,
            ast::Type::Array { elem_type, .. } => {
                let inner = self.lower_type(elem_type, global_ctx);
                hir::Type::Array(inner)
            }
            ast::Type::Func { input, output, .. } => {
                let input = self.lower_type(input, global_ctx);
                let output = self.lower_type(output, global_ctx);
                hir::Type::Function { input, output }
            }
            ast::Type::Path { segments, span } => 'path: {
                if segments.len() != 1 {
                    self.insert_error(SemanticError::QualifiedTypeNotSupported { span: *span });
                    break 'path hir::Type::Error;
                }
                match lower_str_to_type(&segments[0].ident(), span) {
                    Ok(inner) => inner,
                    Err(err) => {
                        self.insert_error(err);
                        hir::Type::Error
                    }
                }
            }
        };
        global_ctx.types.intern(&lowered)
    }

    /// Given the type declaration parameters of some symbol, generate a symbol builder.
    ///
    /// ```cm
    /// foo :: Int -> Int
    /// ```
    fn lower_type_declaration(
        &mut self,
        declared_type: &ast::Type,
        declared_name: &ast::Ident,
        global_context: &mut hir::GlobalContext,
    ) -> hir::SymbolBuilder {
        let lowered_ty = self.lower_type(declared_type, global_context);
        let name_span = declared_name.span();
        let lowered_id = self.lower_ident(declared_name);
        let err_expr =
            self.lower_expression(&ast::Expression::Error(Span::from(0..0)), global_context);
        let symbol_kind = SymbolKind::Defn {
            span_type: name_span,
            span_decl: Span::from(0..0),
            declaration: SymbolDec {
                inputs: Box::new([]),
                body: err_expr,
            },
        };
        let reserved = self.symbols.push(Symbol {
            ty: lowered_ty,
            name: lowered_id,
            kind: symbol_kind,
        });
        let _ = self.insert_symbol_to_current_scope(lowered_id, reserved, name_span);
        hir::SymbolBuilder::new(lowered_ty, lowered_id, name_span, None, None, reserved)
    }

    fn finish_builder(&mut self, builder: SymbolBuilder) -> Result<ids::SymbolId, SemanticError> {
        let symbol = self.symbols.get_unchecked_mut(builder.reserved_spot);
        if let SymbolKind::Defn {
            span_decl,
            declaration,
            ..
        } = &mut symbol.kind
        {
            *span_decl = builder.span_name_decl.unwrap();
            *declaration = builder.symbol_declaration.unwrap();
            return Ok(builder.reserved_spot);
        }
        Err(SemanticError::InternalError {
            msg: "finish_builder should only be used to build calamars values",
            span: symbol.span(),
        })
    }

    fn lower_binding_declaration(
        &mut self,
        name: &ast::Ident,
        params: &Vec<ast::Ident>,
        body: &ast::Expression,
        mut builder: SymbolBuilder,
        global_ctx: &mut hir::GlobalContext,
    ) -> Result<ids::SymbolId, SemanticError> {
        // Lower the inputs to symbols
        let (input_types, _) = take_inputs(builder.ty, params.len(), global_ctx, name.span())?;
        let inputs: Box<[ids::SymbolId]> = params
            .iter()
            .zip(input_types)
            .map(|(ident, ty)| {
                let name = self.lower_ident(ident);
                let symbol = Symbol {
                    ty,
                    name,
                    kind: SymbolKind::Param { span: ident.span() },
                };
                self.symbols.push(symbol)
            })
            .collect();

        self.push_scope();
        // Add inputs to the functions scope
        for symbol_id in inputs.iter() {
            let symbol = self.symbols.get_unchecked(*symbol_id);
            let _ = self.insert_symbol_to_current_scope(symbol.name, *symbol_id, symbol.span());
        }
        let body = self.lower_expression(body, global_ctx);
        self.pop_scope();

        builder.span_name_decl = Some(name.span());
        builder.symbol_declaration = Some(SymbolDec { inputs, body });
        self.finish_builder(builder)
    }

    /// Given some module, lower it and return any errors that were generated along the way
    pub fn lower_module(
        mut self,
        module: &ast::Module,
        id: ids::FileId,
        name: String,
        global_ctx: &mut hir::GlobalContext,
    ) -> (hir::Module, Vec<SemanticError>) {
        let mut roots = vec![];

        // Generate the builders from the type declarations
        let mut builders = hashbrown::HashMap::new();
        for declaration in &module.items {
            let Declaration::TypeSignature { name, dtype, .. } = declaration else {
                continue;
            };
            let sb = self.lower_type_declaration(dtype, name, global_ctx);
            let old = builders.insert(sb.ident_id(), sb);
            if let Some(original) = old {
                self.insert_error(SemanticError::Redeclaration {
                    original_span: original.span_name_type,
                    redec_span: name.span(),
                })
            }
        }

        // Attach the body to the symbol builders
        for declaration in &module.items {
            let Declaration::Binding { name, params, body } = declaration else {
                continue;
            };
            let name_id = self.lower_ident(name);
            let builder = builders.remove(&name_id);

            match builder {
                Some(builder) => {
                    match self.lower_binding_declaration(name, params, body, builder, global_ctx) {
                        Ok(symbol_id) => roots.push(symbol_id),
                        Err(err) => self.insert_error(err),
                    }
                }
                None => self.insert_error(SemanticError::TypeMissing {
                    for_identifier: name.span(),
                }),
            }
        }

        // these are the ones that had a type declaration but no body
        for (_, builder) in builders {
            self.insert_error(SemanticError::MissingDeclaration {
                name: builder.name,
                type_declaration: builder.span_name_type,
            });
        }

        (
            hir::Module {
                id,
                name,
                idents: self.identifiers,
                symbols: self.symbols,
                exprs: self.expressions,
                roots: roots.into_boxed_slice(),
                expression_types: hashbrown::HashMap::new(),
            },
            self.diag_err,
        )
    }
}
