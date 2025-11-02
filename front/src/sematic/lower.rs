//! Lower AST to HIR

use calamars_core::ids::{self, SymbolId};

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, Const, Expr, Symbol},
    },
    syntax::{ast, span::Span},
};

pub struct LowererContext {
    types: hir::TypeArena,
    const_str: hir::ConstantStringArena,
    identifiers: hir::IdentArena,
    expressions: hir::ExpressionArena,
    symbols: hir::SymbolArena,
    /// Scopes are used for shadowing. When in the same scope we do not allow shadowing.
    scopes: Vec<hashbrown::HashMap<ids::IdentId, SymbolId>>,

    /// Errors, if any are present, we cannot proceed with compilation
    diag_err: Vec<SemanticError>,
    /// Warnings, if any are presenet, we can still proceed with compilation
    diag_war: Vec<()>,
}

impl LowererContext {
    fn insert_error(&mut self, err: SemanticError) {
        self.diag_err.push(err);
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

    fn type_lower(&mut self, ty: &ast::Type) -> ids::TypeId {
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
                    .map(|t| self.type_lower(t))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let output = self.type_lower(output.as_ref());
                hir::Type::Function { input, output }
            }
            ast::Type::Array { elem_type, .. } => {
                hir::Type::Array(self.type_lower(elem_type.as_ref()))
            }
        };
        self.types.intern(&ty)
    }

    pub fn module(&mut self, module: &ast::Module) -> Box<[ids::SymbolId]> {
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

        let (mut input_idents, mut input_spans) = (vec![], vec![]);
        for i in def.input_idents() {
            let name = i.ident().to_string();
            input_idents.push(self.identifiers.intern(&name));
            input_spans.push(i.span());
        }

        let ty = self.type_lower(def.fntype());
        let kind = hir::SymbolKind::FunctionUndeclared;
        let symbol = Symbol::new(kind, ty, name, def.name_span(), def.span());
        self.symbols.push(symbol)
    }

    fn bind_declaration(&mut self, bind: &ast::Binding) -> SymbolId {
        let str_name = bind.vname.ident().to_string();
        let name = self.identifiers.intern(&str_name);
        let ty = self.type_lower(&bind.vtype);
        let kind = hir::SymbolKind::VariableUndeclared {
            mutable: bind.mutable,
        };
        let symbol = Symbol::new(kind, ty, name, bind.name_span(), bind.span());
        self.symbols.push(symbol)
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
                    _ => break 'binop_case Expr::Err,
                };
                let lhs = self.expression(&binary_op.lhs());
                let rhs = self.expression(&binary_op.rhs());
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
}
