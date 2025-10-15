use std::collections::HashMap;

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct TypeId(usize);

#[rustfmt::skip]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Type {
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
                format!("({}) -> {}", inp, out)
            }
        }
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
}
