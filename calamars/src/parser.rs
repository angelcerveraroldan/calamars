//! Parser for Calamars

// TOOD: Fix later, import what you need only
use chumsky::*;

enum ClPrimitive {
    Integer,
    Float,
    String,
    Boolean,
}

enum BinaryOp {
    Add,
    Sub,
    Times,
    Div,
    Neg,
}

enum Expression {
    Literal(ClPrimitive),
}
