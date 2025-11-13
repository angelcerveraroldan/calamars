use paste::paste;

/// Generate Ids for given identifiers.
macro_rules! id_gen {
    ($($name:ident),*$(,)?) => {
        $(
            paste! {
                #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
                pub struct [<$name Id>](usize);
                impl [<$name Id>] {
                    pub fn inner(&self) -> usize { self.0 }
                }
                impl From<usize> for [<$name Id>] { fn from(u: usize) -> Self { Self(u) } }
                impl crate::Identifier for [<$name Id>] {
                    fn inner_id(&self) -> usize { self.inner() }
                }
            }
        )*
    };
}

id_gen!(Type, Symbol, Expression, Ident, String, File);
