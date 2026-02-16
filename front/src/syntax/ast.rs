use crate::syntax::span::Span;

/// An identifier
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    /// The identifier
    ident: String,
    /// Where it is found in the source
    span: Span,
}

impl Ident {
    pub fn new(ident: String, span: Span) -> Self {
        Self { ident, span }
    }

    pub fn ident(&self) -> &str {
        &self.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

/// A Calamars module / file
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub imports: Vec<Import>,
    pub items: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    idents: Vec<Ident>,
    total_span: Span,
}

impl Import {
    pub fn new(idents: Vec<Ident>, total_span: Span) -> Self {
        Self { idents, total_span }
    }

    pub fn span(&self) -> Span {
        self.total_span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Declaration(Declaration),
    Expression(Expression),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Declaration(cl_declaration) => cl_declaration.span(),
            Item::Expression(cl_expression) => cl_expression.span(),
        }
    }
}

/// Calamars Base Type kind and value
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralKind {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<Literal>),
}

/// Calamars Base Type, along with its span in the source
#[derive(Debug, Clone, PartialEq)]
pub struct Literal {
    kind: LiteralKind,
    span: Span,
}

impl Literal {
    pub fn new(kind: LiteralKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &LiteralKind {
        &self.kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

impl From<Literal> for Expression {
    fn from(value: Literal) -> Self {
        Expression::Literal(value)
    }
}

/// Types for Calamars
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// The parser will generate this error when it cannot parse the type
    Error(Span),
    Unit(Span),
    /// Basic / standard types such as Int, String, Char, Real, ...
    /// as well as types that require many segments, such as people.Person
    Path {
        segments: Vec<Ident>,
        span: Span,
    },
    /// An array of some type such as [Int]
    Array {
        elem_type: Box<Self>,
        span: Span,
    },
    /// A function (I1, I2, I3, ...) -> (O1, O2, O3, ...)
    Func {
        input: Box<Self>,
        output: Box<Self>,
        span: Span,
    },
}

impl Type {
    pub fn new_path(segments: Vec<Ident>, span: Span) -> Self {
        Self::Path { segments, span }
    }

    pub fn span(&self) -> Span {
        match self {
            Type::Path { span, .. }
            | Type::Array { span, .. }
            | Type::Error(span)
            | Type::Func { span, .. }
            | Type::Unit(span) => *span,
        }
    }

    pub fn is_err(&self) -> bool {
        matches!(self, Type::Error(_))
    }

    pub fn is_ok(&self) -> bool {
        !self.is_err()
    }
}

/// All types of possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Error(Span),

    Literal(Literal),
    Identifier(Ident),

    UnaryOp(UnaryOp),
    BinaryOp(BinaryOp),

    IfStm(IfStm),
    Apply(Apply),

    Block(CompoundExpression),
}

impl Expression {
    pub fn span(&self) -> Span {
        match self {
            Expression::Literal(cl_literal) => cl_literal.span(),
            Expression::Identifier(ident) => ident.span(),
            Expression::UnaryOp(cl_unary_op) => cl_unary_op.span(),
            Expression::BinaryOp(cl_binary_op) => cl_binary_op.span(),
            Expression::IfStm(if_stm) => if_stm.span(),
            Expression::Apply(func_call) => func_call.span(),
            Expression::Block(cl_compound_expression) => cl_compound_expression.total_span(),
            // I am not a big fan of this ...
            Expression::Error(span) => *span,
        }
    }

    /// If something is retuend, span to the expression being returned
    ///
    /// If the expression has Unit return type, then return nothing
    ///
    /// TODO: If statements are not handled well due to having two arms. This will get worse with
    /// match statements that have `n` arms.
    pub fn returning_span(&self) -> Option<Span> {
        if let Expression::Block(cl_compound_expression) = self {
            cl_compound_expression.return_span()
        } else {
            Some(self.span())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,      // +
    Sub,      // -
    Times,    // *
    Pow,      // ^
    Div,      // /
    Concat,   // ++
    Less,     // <
    Greater,  // >
    Geq,      // >=
    Leq,      // <=
    EqEq,     // ==
    NotEqual, // !=
    Mod,      // %

    Or,  // or
    Xor, // xor
    And, // and
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOp {
    operator: UnaryOperator,
    on: Box<Expression>,
    span: Span,
}

impl From<UnaryOp> for Expression {
    fn from(value: UnaryOp) -> Self {
        Expression::UnaryOp(value)
    }
}

impl UnaryOp {
    pub fn new(operator: UnaryOperator, on: Box<Expression>, span: Span) -> Self {
        Self { operator, on, span }
    }

    pub fn operator(&self) -> &UnaryOperator {
        &self.operator
    }

    pub fn inner_exp(&self) -> &Box<Expression> {
        &self.on
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    operator: BinaryOperator,
    left: Box<Expression>,
    right: Box<Expression>,
    span: Span,
}

impl From<BinaryOp> for Expression {
    fn from(value: BinaryOp) -> Self {
        Expression::BinaryOp(value)
    }
}

impl BinaryOp {
    pub fn new(
        operator: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
        span: Span,
    ) -> Self {
        Self {
            operator,
            left,
            right,
            span,
        }
    }

    pub fn operator(&self) -> &BinaryOperator {
        &self.operator
    }

    pub fn lhs(&self) -> &Box<Expression> {
        &self.left
    }

    pub fn rhs(&self) -> &Box<Expression> {
        &self.right
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfStm {
    predicate: Box<Expression>,
    then: Box<Expression>,
    otherwise: Box<Expression>,

    span: Span,
}

impl IfStm {
    pub fn new(
        predicate: Box<Expression>,
        then: Box<Expression>,
        otherwise: Box<Expression>,
        span: Span,
    ) -> Self {
        Self {
            predicate,
            then,
            otherwise,
            span,
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn pred_span(&self) -> Span {
        self.predicate.span()
    }

    pub fn then_span(&self) -> Span {
        self.then.span()
    }

    pub fn else_span(&self) -> Span {
        self.otherwise.span()
    }

    pub fn then_expr(&self) -> &Box<Expression> {
        &self.then
    }

    pub fn else_expr(&self) -> &Box<Expression> {
        &self.otherwise
    }

    pub fn pred(&self) -> &Box<Expression> {
        &self.predicate
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Apply {
    func: Box<Expression>,
    input: Box<Expression>,
    span: Span,
}

impl Apply {
    pub fn new(func: Expression, input: Expression, span: Span) -> Self {
        Self {
            func: func.into(),
            input: input.into(),
            span,
        }
    }

    pub fn callable(&self) -> &Box<Expression> {
        &self.func
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn callable_span(&self) -> Span {
        self.func.span()
    }
}

/// An expression in the form
///
/// {
///    (<cl_item>)*;
///    (<cl_expression>)?
/// }
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundExpression {
    pub items: Vec<Item>,
    pub final_expr: Option<Box<Expression>>,

    span: Span,
}

impl CompoundExpression {
    pub fn new(items: Vec<Item>, final_expr: Option<Box<Expression>>, span: Span) -> Self {
        Self {
            items,
            final_expr,
            span,
        }
    }

    pub fn total_span(&self) -> Span {
        self.span
    }

    pub fn return_span(&self) -> Option<Span> {
        self.final_expr.clone().map(|x| x.span())
    }
}

// DECLARATIONS

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    TypeSignature {
        docs: Option<String>,
        name: Ident,
        dtype: Type,
    },
    Binding {
        name: Ident,
        params: Vec<Ident>,
        body: Expression,
    },
}

impl Declaration {
    pub fn span(&self) -> Span {
        match self {
            Declaration::TypeSignature { dtype, .. } => dtype.span(),
            Declaration::Binding { body, .. } => body.span(),
        }
    }

    pub fn name_span(&self) -> Span {
        match self {
            Declaration::TypeSignature { name, .. } | Declaration::Binding { name, .. } => {
                name.span()
            }
        }
    }
}
