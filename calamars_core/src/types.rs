use crate::MaybeErr;
use crate::ids;

pub type TypeArena = crate::InternArena<Type, ids::TypeId>;

impl Default for TypeArena {
    fn default() -> Self {
        let mut ta = TypeArena::new_checked();
        ta.intern(&Type::Error);
        ta.intern(&Type::Unit);
        ta.intern(&Type::Integer);
        ta.intern(&Type::Float);
        ta.intern(&Type::Boolean);
        ta.intern(&Type::String);
        ta.intern(&Type::Char);
        ta
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Type {
    /// Among other things, this is used for when we expected a type, but found ClType::None
    ///
    /// We can:
    /// 1. Try to recover (type inference)
    /// 2. Throw a detailed error
    Error,

    // Primitives
    Integer,
    Float,
    Boolean,
    String,
    Char,
    Unit,

    Array(ids::TypeId),
    Function {
        input: ids::TypeId,
        output: ids::TypeId,
    },
    Structure(ids::DStructId),
}

impl Type {
    pub fn function_input(&self) -> &ids::TypeId {
        if let Type::Function { input, .. } = self {
            return input;
        }
        unreachable!("Make sure to only call this on functions!")
    }

    pub fn function_output(&self) -> ids::TypeId {
        if let Type::Function { output, .. } = self {
            return *output;
        }
        unreachable!("Make sure to only call this on functions!")
    }
}

impl MaybeErr for Type {
    const ERR: Self = Type::Error;
}

pub fn type_id_stringify(arena: &TypeArena, id: ids::TypeId) -> String {
    let ty = arena.get_unchecked(id);

    match ty {
        Type::Error => "Error".into(),
        Type::Integer => "Int".into(),
        Type::Float => "Float".into(),
        Type::Boolean => "Bool".into(),
        Type::String => "String".into(),
        Type::Char => "Char".into(),
        Type::Unit => "Unit".into(),
        Type::Array(tid) => format!("[{}]", type_id_stringify(arena, *tid)),
        Type::Function { input, output } => {
            let inp = type_id_stringify(arena, *input);
            let out = type_id_stringify(arena, *output);
            format!("{inp} -> ({out})")
        }
        Type::Structure(_id) => format!("Struct (TODO: name)"),
    }
}
