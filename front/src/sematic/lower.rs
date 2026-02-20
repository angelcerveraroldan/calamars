//! Lower AST to HIR

use calamars_core::ids;

use crate::{
    sematic::{
        error::SemanticError,
        hir::{self, SymbolBuilder, SymbolDec},
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

/// Lower AST to HIR in two passes:
/// - Pass A (`declare_symbols_pass`) walks declarations, creates symbols, and fills scopes so uses can
///   resolve regardless of order.
/// - Pass B (`attach_body_declaration`) lowers bodies/expressions now that all symbols exist.
///
/// This prevents order-dependent resolution errors (e.g., calling a function declared later).
pub struct HirBuilder {
    pub identifiers: hir::IdentArena,
    pub expressions: hir::ExpressionArena,
    pub symbols: hir::SymbolArena,

    /// Scopes are used for shadowing. When in the same scope we do not allow shadowing.
    scopes: Vec<hashbrown::HashMap<ids::IdentId, ids::SymbolId>>,

    /// Errors, if any are present, we cannot proceed with compilation
    pub diag_err: Vec<SemanticError>,
    /// Warnings, if any are present, we can still proceed with compilation
    diag_war: Vec<()>,
}

impl Default for HirBuilder {
    fn default() -> Self {
        let mut d = Self {
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

    fn lower_ident(&mut self, ident: &ast::Ident) -> ids::IdentId {
        let identifier_name = ident.ident().to_string();
        self.identifiers.intern(&identifier_name)
    }

    fn lower_expression(&mut self, expr: &ast::Expression) -> ids::ExpressionId {
        todo!()
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
        hir::SymbolBuilder::new(lowered_ty, lowered_id, name_span, None, None)
    }

    fn lower_binding_declaration(
        &mut self,
        name: &ast::Ident,
        params: &Vec<ast::Ident>,
        body: &ast::Expression,
        mut builder: SymbolBuilder,
    ) -> Result<hir::Symbol, SemanticError> {
        let inputs: Box<[ids::IdentId]> =
            params.iter().map(|ident| self.lower_ident(ident)).collect();
        let body = self.lower_expression(body);

        builder.span_name_decl = Some(name.span());
        builder.symbol_declaration = Some(SymbolDec { inputs, body });
        builder.finish()
    }

    /// Given some module, lower it and return any errors that were generated along the way
    pub fn lower_module(
        &mut self,
        module: &ast::Module,
        id: ids::FileId,
        name: String,
        global_ctx: &mut hir::GlobalContext,
    ) -> hir::Module {
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

            if let Some(builder) = builder {
                self.lower_binding_declaration(name, params, body, builder);
            } else {
                self.insert_error(SemanticError::TypeMissing {
                    for_identifier: name.span(),
                });
            }
        }

        // these are the ones that had a type declaration but no body
        for (ident, builder) in builders {
            self.insert_error(SemanticError::MissingDeclaration {
                name: builder.name,
                type_declaration: builder.span_name_type,
            });
        }

        todo!()
    }
}
