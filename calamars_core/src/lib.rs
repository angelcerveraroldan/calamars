//! Structures and functions that will be used form the front, to the back end of the language

pub mod ids;

use std::{hash::Hash, marker::PhantomData, ops::Deref};

pub trait Identifier
where
    Self: From<usize> + Copy,
{
    fn inner_id(&self) -> usize;
}

pub trait MaybeErr
where
    Self: PartialEq + Sized,
{
    const ERR: Self;

    fn is_error(&self) -> bool {
        self == &Self::ERR
    }
}

pub trait PushPolicy<T> {
    fn accept(v: &T) -> bool;
}

pub struct NoFilter;
impl<T> PushPolicy<T> for NoFilter {
    fn accept(_: &T) -> bool {
        true
    }
}

pub struct RejectErr;
impl<T: MaybeErr> PushPolicy<T> for RejectErr {
    fn accept(v: &T) -> bool {
        !v.is_error()
    }
}

#[derive(Debug, Default)]
pub struct InternArena<Ty, Id> {
    data: Vec<Ty>,
    map: hashbrown::HashMap<Ty, Id>,
}

#[derive(Debug, Default)]
pub struct PolicyArena<Ty, Id, P: PushPolicy<Ty>> {
    data: Vec<Ty>,
    _pd: PhantomData<(Id, P)>,
}

/// An arena that will push everything given, without any sort of check
pub type UncheckedArena<Ty, Id> = PolicyArena<Ty, Id, NoFilter>;
/// An arena that will push anything except errors
pub type Arena<Ty, Id> = PolicyArena<Ty, Id, RejectErr>;

impl<Ty: MaybeErr, Id: Identifier, P: PushPolicy<Ty>> PolicyArena<Ty, Id, P> {
    fn new() -> Self {
        Self {
            data: vec![Ty::ERR],
            _pd: PhantomData,
        }
    }

    pub fn err_id(&self) -> Id {
        Id::from(0)
    }
}

impl<Ty: MaybeErr + Clone + Hash + Eq, Id: Identifier> InternArena<Ty, Id> {
    fn new() -> Self {
        let mut s = Self {
            data: vec![Ty::ERR],
            map: hashbrown::HashMap::new(),
        };
        s.map.insert(Ty::ERR, Id::from(0));
        s
    }

    pub fn err_id(&self) -> Id {
        Id::from(0)
    }
}

impl<Ty, Id: Identifier, P: PushPolicy<Ty>> PolicyArena<Ty, Id, P> {
    pub fn push(&mut self, ty: Ty) -> Id {
        // Dont push errors
        if !P::accept(&ty) {
            return Id::from(0);
        }
        self.data.push(ty);
        Id::from(self.data.len() - 1)
    }
}

impl<Ty: Hash + Eq + Clone, Id: Identifier> InternArena<Ty, Id> {
    pub fn intern(&mut self, ty: &Ty) -> Id {
        if let Some(id) = self.map.get(ty) {
            return *id;
        }
        self.data.push(ty.clone());
        let id = Id::from(self.data.len() - 1);
        self.map.insert(ty.clone(), id);
        id
    }
}

impl<Ty, Id: Identifier, P: PushPolicy<Ty>> PolicyArena<Ty, Id, P> {
    pub fn get(&self, id: Id) -> Option<&Ty> {
        self.data.get(id.inner_id())
    }

    pub fn get_unchecked(&self, id: Id) -> &Ty {
        &self.data[id.inner_id()]
    }
}

impl<Ty, Id: Identifier> InternArena<Ty, Id> {
    pub fn get(&self, id: Id) -> Option<&Ty> {
        self.data.get(id.inner_id())
    }

    pub fn get_unchecked(&self, id: Id) -> &Ty {
        &self.data[id.inner_id()]
    }
}
