use std::collections::HashMap;

use crate::{
    sematic::{error::SemanticError, types},
    syntax::{
        ast::{self, Ident},
        span::Span,
    },
};

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct TypeId(usize);

#[rustfmt::skip]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Type {
    /// Among other things, this is used for when we expected a type, but found ClType::None
    ///
    /// We can:
    /// 1. Try to recover (type inference)
    /// 2. Throw a detailed error
    Error,

    // Primitives
    Integer, Float, Boolean, String, Char, Unit,

    Array(TypeId),

    Function {
        input: Vec<TypeId>, output: TypeId,
    }

    // TODO: (later)
    // - Custom types
    // - Tuples
    // - structs
    // - enums
    // - traits and generics
}

#[derive(Debug)]
pub struct TypeArena {
    // An arena containing all types
    arena: Vec<Type>,
    /// Given a type, return its id. The id will be the index in the arena vector.
    index: HashMap<Type, TypeId>,
}

impl Default for TypeArena {
    fn default() -> Self {
        let mut arena = TypeArena {
            arena: vec![],
            index: HashMap::new(),
        };
        // Add the default errors to the arena
        arena.intern(Type::Error);

        // Add the default types to the arena
        arena.intern(Type::Integer);
        arena.intern(Type::Float);
        arena.intern(Type::Boolean);
        arena.intern(Type::String);
        arena.intern(Type::Char);
        arena.intern(Type::Unit);
        arena
    }
}

impl TypeArena {
    /// Pretty print types
    pub fn as_string(&self, type_id: TypeId) -> String {
        let ty = &self.arena[type_id.0];
        match ty {
            Type::Error => "Error".into(),
            Type::Integer => "int".into(),
            Type::Float => "float".into(),
            Type::Boolean => "bool".into(),
            Type::String => "str".into(),
            Type::Char => "char".into(),
            Type::Unit => "()".into(),
            Type::Array(tid) => format!("[{}]", self.as_string(*tid)),
            Type::Function { input, output } => {
                let inp = input
                    .iter()
                    .map(|x| self.as_string(*x))
                    .collect::<Vec<_>>()
                    .join(", ");
                let out = self.as_string(*output);
                format!("fn({}) -> ({})", inp, out)
            }
        }
    }

    pub fn many_types_as_str(&self, type_ids: Vec<TypeId>) -> String {
        type_ids
            .into_iter()
            .map(|ty| self.as_string(ty))
            .collect::<Vec<String>>()
            .join(" or ")
    }

    /// If already in the type arena, then return the types id. If this type is new to the arena,
    /// then add it and return its id.
    pub fn intern(&mut self, ty: Type) -> TypeId {
        if let Some(&id) = self.index.get(&ty) {
            return id;
        }
        let id = TypeId(self.arena.len());
        self.arena.push(ty.clone());
        self.index.insert(ty, id);
        id
    }

    pub fn get(&self, id: TypeId) -> Option<&Type> {
        self.arena.get(id.0)
    }

    pub fn unchecked_get(&self, id: TypeId) -> &Type {
        &self.arena[id.0]
    }

    /// Convert a type from the ast to a semantic type
    pub fn intern_cltype(&mut self, ast_type: &ast::ClType) -> Result<TypeId, SemanticError> {
        match ast_type {
            ast::ClType::Path { segments, span } => match &segments[..] {
                [ident] => match ident.ident() {
                    "Int" => Ok(self.intern(Type::Integer)),
                    "Float" => Ok(self.intern(Type::Float)),
                    "Bool" => Ok(self.intern(Type::Boolean)),
                    "String" => Ok(self.intern(Type::String)),
                    "Char" => Ok(self.intern(Type::Char)),
                    "()" => Ok(self.intern(Type::Unit)),
                    otherwise => Err(SemanticError::TypeNotFound {
                        type_name: otherwise.into(),
                        span: *span,
                    }),
                },
                _ => Err(SemanticError::QualifiedTypeNotSupported { span: *span }),
            },
            ast::ClType::Array { elem_type, span } => {
                let inner_type = self.intern_cltype(&*elem_type)?;
                Ok(self.intern(Type::Array(inner_type)))
            }
            ast::ClType::Func { inputs, output, .. } => {
                let input = inputs
                    .iter()
                    .map(|ty| match ty {
                        Some(t) => self.intern_cltype(t),
                        None => Ok(self.intern(Type::Error)),
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // If no output is defined for the function, the default will be Unit
                let output = match output.as_ref() {
                    Some(out) => self.intern_cltype(&out)?,
                    None => self.intern(Type::Unit),
                };

                let func = Type::Function { input, output };
                Ok(self.intern(func))
            }
        }
    }

    fn get_unchecked(&self, ty: Type) -> TypeId {
        *self.index.get(&ty).unwrap()
    }

    pub fn err(&self) -> TypeId {
        self.get_unchecked(Type::Error)
    }

    pub fn int(&self) -> TypeId {
        self.get_unchecked(Type::Integer)
    }

    pub fn float(&self) -> TypeId {
        self.get_unchecked(Type::Float)
    }

    pub fn bool(&self) -> TypeId {
        self.get_unchecked(Type::Boolean)
    }

    pub fn char(&self) -> TypeId {
        self.get_unchecked(Type::Char)
    }

    pub fn string(&self) -> TypeId {
        self.get_unchecked(Type::String)
    }

    pub fn unit(&self) -> TypeId {
        self.get_unchecked(Type::Unit)
    }

    /// Given a functions type id, return the type_id of its output
    ///
    /// Given a fn with signature Int -> Int, this function shuold return Some(typeid of Int)
    pub fn fn_return_typeid(&self, fn_id: TypeId) -> Option<TypeId> {
        self.get(fn_id)
            .map(|fn_ty| match fn_ty {
                Type::Function { output, .. } => Some(*output),
                _ => None,
            })
            .flatten()
    }
}

#[cfg(test)]
mod test_types_sem {
    use chumsky::Parser;

    use crate::{
        parser::{declaration::parse_cldeclaration, parse_cl_item},
        syntax::token::Token,
    };

    use super::*;

    fn make_arena() -> TypeArena {
        TypeArena::default()
    }

    #[test]
    fn test_cltype_var_insertion() {
        let mut arena = make_arena();

        // Make a small tree from source code
        let line = "var x: Int = 2;";
        let stream = Token::tokens_spanned_stream(&line);
        let out = parse_cl_item().parse(stream).unwrap();
        let binding = match out.get_dec() {
            ast::ClDeclaration::Binding(cl_binding) => cl_binding,
            ast::ClDeclaration::Function(cl_func_dec) => panic!("this should not be a fn..."),
        };
        let id = arena
            .intern_cltype(binding.vtype.as_ref().unwrap())
            .unwrap();
        let ty = arena.get(id).unwrap();
        assert_eq!(*ty, Type::Integer);
    }

    #[test]
    fn test_cltype_lambda_insertion() {
        let mut arena = make_arena();

        let integer = arena.intern(Type::Integer);

        /// This makes no sense semantically, as the type is wrong, but we will not be running the
        /// check. Just making sure that the lambda type parses.
        let line = "var foo: Int -> Int = 2;";
        let stream = Token::tokens_spanned_stream(&line);
        let out = parse_cl_item().parse(stream).unwrap();
        let binding = match out.get_dec() {
            ast::ClDeclaration::Binding(cl_binding) => cl_binding,
            ast::ClDeclaration::Function(cl_func_dec) => panic!("this should not be a fn..."),
        };

        let id = arena
            .intern_cltype(binding.vtype.as_ref().unwrap())
            .unwrap();
        let ty = arena.get(id).unwrap();

        assert_eq!(
            *ty,
            Type::Function {
                input: vec![integer],
                output: integer,
            }
        );
    }
}
