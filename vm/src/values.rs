use std::ops::{BitAnd, BitOr, BitXor};

use crate::errors::{VError, VResult};

/// Stack allocated values
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Char(char),

    Empty,
}

macro_rules! binfuncs {
    (closed $name:ident, $e:expr, $($id:ident),*) => {
	pub fn $name(&self, other: &Value) -> VResult<Value> {
	    match (self, other) {
		$((Value::$id(a), Value::$id(b)) => Ok(Value::$id($e(a, b))),)*
		_ => Err(VError::TypeMismatchBinary {
                lhs: self.type_name(),
                rhs: other.type_name(),
            }),
	    }
	}
    };
    (cmp $name:ident, $e:expr, $($id:ident),*) => {
	pub fn $name(&self, other: &Value) -> VResult<Value> {
	    match (self, other) {
		$((Value::$id(a), Value::$id(b)) => Ok(Value::Boolean($e(a, b))),)*
		_ => Err(VError::TypeMismatchBinary {
                lhs: self.type_name(),
                rhs: other.type_name(),
            }),
	    }
	}
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Boolean(_) => "bool",
            Value::Char(_) => "char",
            Value::Empty => "empty",
        }
    }

    pub fn negate(&self) -> VResult<Value> {
        match self {
            Value::Integer(i) => Ok(Value::Integer(-i)),
            Value::Boolean(b) => Ok(Value::Boolean(!b)),
            _ => Err(VError::TypeMismatchUnary {
                op: "negate",
                found: self.type_name(),
            }),
        }
    }

    binfuncs!(closed add, |a,b| a + b, Integer, Float);
    binfuncs!(closed sub, |a,b| a - b, Integer, Float);
    binfuncs!(closed mul, |a,b| a * b, Integer, Float);
    binfuncs!(closed div, |a,b| a / b, Integer, Float);
    binfuncs!(closed modulus, |a,b| a % b, Integer, Float);

    binfuncs!(cmp g, |a,b| a >  b, Integer, Float);
    binfuncs!(cmp l, |a,b| a <  b, Integer, Float);
    binfuncs!(cmp e, |a,b| a == b, Integer, Float, Boolean);
    binfuncs!(cmp ge, |a,b| a >= b, Integer, Float);
    binfuncs!(cmp le, |a,b| a <= b, Integer, Float);

    pub fn xor(&self, other: &Value) -> VResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.bitxor(b))),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a ^ *b)),
            _ => Err(VError::TypeMismatchBinary {
                lhs: self.type_name(),
                rhs: other.type_name(),
            }),
        }
    }

    pub fn or(&self, other: &Value) -> VResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.bitor(b))),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a || *b)),
            _ => Err(VError::TypeMismatchBinary {
                lhs: self.type_name(),
                rhs: other.type_name(),
            }),
        }
    }

    pub fn and(&self, other: &Value) -> VResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.bitand(b))),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a && *b)),
            _ => Err(VError::TypeMismatchBinary {
                lhs: self.type_name(),
                rhs: other.type_name(),
            }),
        }
    }
}

#[test]
fn value_ops_cover_basics() {
    //  yes this is ugly, but we can deal with it, it will all be ok.
    assert_eq!(
        Value::Integer(5).add(&Value::Integer(-1)),
        Ok(Value::Integer(4))
    );
    assert_eq!(
        Value::Integer(7).sub(&Value::Integer(2)),
        Ok(Value::Integer(5))
    );
    assert_eq!(
        Value::Integer(3).mul(&Value::Integer(4)),
        Ok(Value::Integer(12))
    );
    assert_eq!(
        Value::Integer(8).div(&Value::Integer(2)),
        Ok(Value::Integer(4))
    );
    assert_eq!(
        Value::Integer(9).modulus(&Value::Integer(4)),
        Ok(Value::Integer(1))
    );
    assert_eq!(
        Value::Float(1.5).add(&Value::Float(2.0)),
        Ok(Value::Float(3.5))
    );
    assert_eq!(
        Value::Float(5.0).div(&Value::Float(2.0)),
        Ok(Value::Float(2.5))
    );
    assert_eq!(
        Value::Integer(6).xor(&Value::Integer(3)),
        Ok(Value::Integer(5))
    );
    assert_eq!(
        Value::Integer(6).or(&Value::Integer(3)),
        Ok(Value::Integer(7))
    );
    assert_eq!(
        Value::Integer(6).and(&Value::Integer(3)),
        Ok(Value::Integer(2))
    );
    assert_eq!(
        Value::Boolean(true).xor(&Value::Boolean(false)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Boolean(true).or(&Value::Boolean(false)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Boolean(true).and(&Value::Boolean(false)),
        Ok(Value::Boolean(false))
    );
    assert_eq!(
        Value::Integer(3).g(&Value::Integer(2)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Integer(3).l(&Value::Integer(2)),
        Ok(Value::Boolean(false))
    );
    assert_eq!(
        Value::Integer(3).e(&Value::Integer(3)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Integer(3).ge(&Value::Integer(3)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Integer(2).le(&Value::Integer(3)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Float(2.0).g(&Value::Float(1.0)),
        Ok(Value::Boolean(true))
    );
    assert_eq!(
        Value::Float(2.0).e(&Value::Float(2.0)),
        Ok(Value::Boolean(true))
    );
}
