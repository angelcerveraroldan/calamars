//! Lower AST to HIR

use calamars_core::ids::{self, SymbolId};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, Const, Expr, Symbol, default_typearena},
    },
    syntax::{
        ast::{self, CompoundExpression},
        span::Span,
    },
};

/// Lower AST to HIR in two passes:
/// - Pass A (`module_pass_a`) walks declarations, creates symbols, and fills scopes so uses can
///   resolve regardless of order.
/// - Pass B (`attach_body_declaration`) lowers bodies/expressions now that all symbols exist.
///
/// This prevents order-dependent resolution errors (e.g., calling a function declared later).
pub struct HirBuilder {
    pub types: hir::TypeArena,
    pub const_str: hir::ConstantStringArena,
    pub identifiers: hir::IdentArena,
    pub expressions: hir::ExpressionArena,
    pub symbols: hir::SymbolArena,

    /// Scopes are used for shadowing. When in the same scope we do not allow shadowing.
    scopes: Vec<hashbrown::HashMap<ids::IdentId, SymbolId>>,

    /// Errors, if any are present, we cannot proceed with compilation
    pub diag_err: Vec<SemanticError>,
    /// Warnings, if any are presenet, we can still proceed with compilation
    diag_war: Vec<()>,
}

impl Default for HirBuilder {
    fn default() -> Self {
        let mut d = Self {
            types: default_typearena(),
            const_str: hir::ConstantStringArena::new_unchecked(),
            identifiers: hir::IdentArena::new_unchecked(),
            expressions: hir::ExpressionArena::new_checked(),
            symbols: hir::SymbolArena::new_unchecked(),
            scopes: vec![],
            diag_err: vec![],
            diag_war: vec![],
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

    /// Add a new symbol to the symbol arena, and to the current scope
    ///
    /// This will check for re-declarations. If there is a re-declaration, then ont
    fn insert_symbol(&mut self, symbol: Symbol) -> ids::SymbolId {
        let symbol_ident_id = symbol.ident_id();
        let symbol_span = symbol.name_span();
        let sid = self.symbols.push(symbol);
        let scope = self.scopes.last_mut().expect("Scopes can never be empty!");

        // Here we will overwirte the map, from now on, when someone references this symbol, they
        // will be referring to this new one. If you have the old SymbolId, you can still access
        // the old symbol, as it is NOT delted from the arena.
        let old = scope.insert(symbol_ident_id, sid);

        if let Some(original) = old {
            let osymbol = self.symbols.get_unchecked(original);
            self.insert_error(SemanticError::Redeclaration {
                original_span: osymbol.name_span(),
                redec_span: symbol_span,
            })
        }
        sid
    }

    /// Given a functions declaration detials, generate SymbolIds for the input parameters, but
    /// dont add them to the current scope.
    fn lower_params_to_symbols(
        &mut self,
        idents: &Vec<ast::Ident>,
        ty_id: ids::TypeId,
    ) -> Box<[SymbolId]> {
        if ty_id == self.types.err_id() {
            return [].into();
        }

        let input_tys = self.types.get_unchecked(ty_id).function_input();

        let mut v = vec![];
        for (ident, ty_id) in idents.iter().zip(input_tys) {
            let ident_id = self.identifiers.intern(&ident.ident().to_owned());
            let symbol = Symbol::new(
                hir::SymbolKind::Parameter,
                *ty_id,
                ident_id,
                ident.span(),
                ident.span(),
            );
            v.push(self.symbols.push(symbol));
        }
        v.into()
    }

    fn resolve(&self, s: ids::IdentId, usage: Span) -> Result<SymbolId, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(&s) {
                return Ok(*id);
            }
        }

        // We know that the identifer string must exist (we must have found it to call this
        // function)
        let name = self.identifiers.get_unchecked(s).clone();
        Err(SemanticError::IdentNotFound { name, span: usage })
    }

    fn lower_type(&mut self, ty: &ast::Type) -> ids::TypeId {
        let ty = match ty {
            ast::Type::Error => hir::Type::Error,
            ast::Type::Unit => hir::Type::Unit,
            ast::Type::Path { segments, span } => 'path: {
                if segments.len() != 1 {
                    self.insert_error(SemanticError::QualifiedTypeNotSupported { span: *span });
                    break 'path hir::Type::Error;
                }

                let name = segments[0].ident();
                match name {
                    "Int" | "Integer" => hir::Type::Integer,
                    "Float" | "Real" => hir::Type::Float,
                    "Bool" | "Boolean" => hir::Type::Boolean,
                    "String" => hir::Type::String,
                    "Char" => hir::Type::Char,
                    "Unit" | "()" => hir::Type::Unit,
                    other => {
                        self.insert_error(SemanticError::TypeNotFound {
                            type_name: other.to_string(),
                            span: *span,
                        });
                        hir::Type::Error
                    }
                }
            }
            ast::Type::Func { inputs, output, .. } => {
                let input = inputs
                    .iter()
                    .map(|t| self.lower_type(t))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let output = self.lower_type(output.as_ref());
                hir::Type::Function { input, output }
            }
            ast::Type::Array { elem_type, .. } => {
                hir::Type::Array(self.lower_type(elem_type.as_ref()))
            }
        };
        self.types.intern(&ty)
    }

    pub fn module(&mut self, module: &ast::Module) -> Box<[SymbolId]> {
        // Insert all of the symbols to the table
        let pass_a: Box<[SymbolId]> = self.module_pass_a(module);

        // Add bodies to the declarations
        for (dec, id) in module.items.iter().zip(&pass_a) {
            self.attach_body_declaration(dec, *id);
        }

        pass_a
    }

    /// Pass A is responsible for adding all of the declaration names to the scope, it will ignore
    /// the bodies of any function or constant for now.
    ///
    /// The following code shuold run, if we checked the body of foo before declaring X, we would
    /// get an undeclared error.
    ///
    /// ```cm
    /// def foo() = {
    ///     print(bar(x))
    /// }
    ///
    /// def bar(x: Int) = x
    ///
    /// val X: Int = 2;
    /// ```
    ///
    /// Because of this, we handle all of the declarations first without
    /// looking at the bodies.
    fn module_pass_a(&mut self, module: &ast::Module) -> Box<[ids::SymbolId]> {
        for import in &module.imports {
            self.insert_error(SemanticError::NotSupported {
                msg: "Imports are not yet supported, sorry :(",
                span: import.span(),
            });
        }

        module
            .items
            .iter()
            .map(|dec| self.declaration(dec))
            .collect()
    }

    fn func_declaration(&mut self, def: &ast::FuncDec) -> SymbolId {
        let name = self.identifiers.intern(def.name());
        let ty = self.lower_type(def.fntype());
        let params = self.lower_params_to_symbols(def.input_idents(), ty);
        let kind = hir::SymbolKind::FunctionUndeclared { params };
        let symbol = Symbol::new(kind, ty, name, def.name_span(), def.span());
        self.insert_symbol(symbol)
    }

    /// Insert a body into a declaration.
    ///
    /// To call this function, you must ensure that the symbol id is valid.
    fn attach_body_declaration(&mut self, declaration: &ast::Declaration, sid: ids::SymbolId) {
        let body = match declaration {
            ast::Declaration::Binding(binding) => binding.assigned.as_ref(),
            ast::Declaration::Function(func_dec) => func_dec.body(),
        };

        let symbol = self.symbols.get_unchecked(sid);
        let skc = symbol.kind.clone();

        if let hir::SymbolKind::FunctionUndeclared { params } = &skc {
            self.push_scope();
            let last = self.scopes.last_mut().expect("We just created a scope!");
            for id in params {
                let param = self.symbols.get_unchecked(*id);
                last.insert(param.ident_id(), *id);
            }
        }

        let expression_id = self.expression(body);
        if matches!(skc, hir::SymbolKind::FunctionUndeclared { .. }) {
            self.pop_scope();
        }

        // This line will need all used symbols to be in scope. This function should be executed
        // strictly after pass_a.
        let symbol = self.symbols.get_unchecked_mut(sid);
        symbol.update_body(expression_id);
    }

    fn bind_declaration(&mut self, bind: &ast::Binding) -> ids::SymbolId {
        let str_name = bind.vname.ident().to_string();
        let name = self.identifiers.intern(&str_name);
        let ty = self.lower_type(&bind.vtype);
        let kind = hir::SymbolKind::VariableUndeclared {
            mutable: bind.mutable,
        };
        let symbol = Symbol::new(kind, ty, name, bind.name_span(), bind.span());
        self.insert_symbol(symbol)
    }

    /// Responsible for adding "headers". This will generate the symbols, but will not look at
    /// their body.
    ///
    /// This is because order of functions should not matter, if function a calls function b in its
    /// body, and function a is declared before function b, checking the body of function a before
    /// decalring function b, would lead to an error.
    fn declaration(&mut self, declaration: &ast::Declaration) -> SymbolId {
        match declaration {
            ast::Declaration::Binding(binding) => self.bind_declaration(binding),
            ast::Declaration::Function(def) => self.func_declaration(def),
        }
    }

    /// Turn an expression into an ExpressionId
    ///
    /// This does not check for the correctness of the expression, but it may return the id of Expr::Err if there was an error generating the expression
    fn expression(&mut self, expr: &ast::Expression) -> ids::ExpressionId {
        let expr = match expr {
            ast::Expression::Literal(literal) => 'literal_case: {
                let constant = match literal.kind() {
                    ast::LiteralKind::Integer(i) => Const::I64(*i),
                    ast::LiteralKind::Boolean(b) => Const::Bool(*b),
                    ast::LiteralKind::String(s) => Const::String(self.const_str.intern(s)),
                    _ => break 'literal_case Expr::Err,
                };
                let span = literal.span();
                Expr::Literal { constant, span }
            }
            ast::Expression::Identifier(ident) => {
                let span = ident.span();
                let name = ident.ident();

                let ident_id = self.identifiers.intern(&name.to_string());
                match self.resolve(ident_id, span) {
                    Ok(id) => Expr::Identifier { id, span },
                    Err(err) => {
                        self.insert_error(err);
                        Expr::Err
                    }
                }
            }
            ast::Expression::BinaryOp(binary_op) => 'binop_case: {
                let operator = match binary_op.operator() {
                    ast::BinaryOperator::Add => hir::BinOp::Add,
                    ast::BinaryOperator::Sub => hir::BinOp::Sub,
                    ast::BinaryOperator::Times => hir::BinOp::Mult,
                    ast::BinaryOperator::Div => hir::BinOp::Div,
                    ast::BinaryOperator::EqEq => hir::BinOp::EqEq,
                    ast::BinaryOperator::Mod => hir::BinOp::Mod,
                    _ => break 'binop_case Expr::Err,
                };
                let lhs = self.expression(binary_op.lhs());
                let rhs = self.expression(binary_op.rhs());
                let span = binary_op.span();
                Expr::BinaryOperation {
                    operator,
                    lhs,
                    rhs,
                    span,
                }
            }
            ast::Expression::FunctionCall(func_call) => {
                let inputs = func_call
                    .params()
                    .iter()
                    .map(|exp| self.expression(exp))
                    .collect();

                let f = self.expression(func_call.callable());

                Expr::Call {
                    f,
                    inputs,
                    span: func_call.span(),
                }
            }
            ast::Expression::IfStm(ifstm) => {
                let pred = ifstm.pred();
                let then = ifstm.then_expr();
                let othr = ifstm.else_expr();

                let predicate = self.expression(pred.as_ref());
                let then = self.expression(then.as_ref());
                let otherwise = self.expression(othr.as_ref());

                Expr::If {
                    predicate,
                    then,
                    otherwise,
                    span: ifstm.span(),
                    pred_span: ifstm.pred_span(),
                    then_span: ifstm.then_span(),
                    othewise_span: ifstm.else_span(),
                }
            }
            ast::Expression::Block(b) => {
                self.insert_error(SemanticError::NotSupported {
                    msg: "Block expressions are not yet supported",
                    span: b.total_span(),
                });
                Expr::Err
            }
            otherwise => {
                self.insert_error(SemanticError::NotSupported {
                    msg: "Expression not yet supported",
                    span: otherwise.span(),
                });
                Expr::Err
            }
        };
        self.expressions.push(expr)
    }

    pub fn lower_module(
        module: &ast::Module,
        id: ids::FileId,
        name: String,
    ) -> (hir::Module, Vec<SemanticError>) {
        let mut lowerer = HirBuilder::default();
        let roots = lowerer.module(module);
        let errs = lowerer.errors().clone();

        (
            hir::Module {
                id,
                name,
                types: lowerer.types,
                const_str: lowerer.const_str,
                idents: lowerer.identifiers,
                symbols: lowerer.symbols,
                exprs: lowerer.expressions,
                roots,
                expression_types: hashbrown::HashMap::new(),
            },
            errs,
        )
    }
}
