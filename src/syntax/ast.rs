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
    Error,
    /// Basic / standard types such as Int, String, Char, Real, ...
    /// as well as types that require many segments, such as people.Person
    Path { segments: Vec<Ident>, span: Span },
    /// An array of some type such as [Int]
    Array { elem_type: Box<Self>, span: Span },
    /// A function (I1, I2, I3, ...) -> (O1, O2, O3, ...)
    Func {
        inputs: Vec<Option<Self>>,
        output: Box<Option<Self>>,
        /// Lambdas have a very clear "full span", where as functions dont
        span: Option<Span>,
    },
}

impl Type {
    pub fn span(&self) -> Option<Span> {
        match self {
            Type::Path { span, .. }
            | Type::Array { span, .. }
            | Type::Func {
                span: Some(span), ..
            } => Some(*span),
            _ => None,
        }
    }
}

/// All types of possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Literal(Literal),
    Identifier(Ident),

    UnaryOp(UnaryOp),
    BinaryOp(BinaryOp),

    IfStm(IfStm),
    FunctionCall(FuncCall),

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
            Expression::FunctionCall(func_call) => func_call.span(),
            Expression::Block(cl_compound_expression) => cl_compound_expression.total_span(),
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
    Geq,      // >=
    Leq,      // <=
    EqEq,     // ==
    NotEqual, // !=

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
pub struct FuncCall {
    func_name: Ident,
    params: Vec<Expression>,
    span: Span,
}

impl FuncCall {
    pub fn new(func_name: Ident, params: Vec<Expression>, span: Span) -> Self {
        Self {
            func_name,
            params,
            span,
        }
    }

    pub fn params(&self) -> &Vec<Expression> {
        &self.params
    }

    pub fn name(&self) -> &str {
        &self.func_name.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.func_name.span()
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
    Binding(Binding),
    Function(FuncDec),
}

impl Declaration {
    pub fn span(&self) -> Span {
        match self {
            Declaration::Binding(cl_binding) => cl_binding.span(),
            Declaration::Function(cl_func_dec) => cl_func_dec.span(),
        }
    }

    pub fn name_span(&self) -> Span {
        match self {
            Declaration::Binding(cl_binding) => cl_binding.name_span(),
            Declaration::Function(cl_func_dec) => cl_func_dec.name_span(),
        }
    }
}

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub vname: Ident,
    pub vtype: Option<Type>,
    pub assigned: Box<Expression>,
    pub mutable: bool,

    span: Span,
}

impl Binding {
    pub fn new(
        vname: Ident,
        vtype: Option<Type>,
        assigned: Box<Expression>,
        mutable: bool,
        span: Span,
    ) -> Self {
        Self {
            vname,
            vtype,
            assigned,
            mutable,
            span,
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.vname.span()
    }

    pub fn type_span(&self) -> Option<Span> {
        self.vtype.clone().map(|ty| ty.span()).flatten()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDec {
    pub doc_comment: Option<String>,

    fname: Ident,
    input_idents: Vec<Ident>,
    functype: Type,
    body: Expression,
    span: Span,
}

impl FuncDec {
    pub fn new(
        fname: Ident,
        inputs: Vec<(Ident, Option<Type>)>,
        out_type: Option<Type>,
        body: Expression,
        span: Span,
        doc_comment: Option<String>,
    ) -> Self {
        let cap = inputs.len();
        let (input_idents, inputs) = inputs.into_iter().fold(
            (Vec::with_capacity(cap), Vec::with_capacity(cap)),
            |(mut idents, mut tys), (ident, ty)| {
                idents.push(ident);
                tys.push(ty);
                (idents, tys)
            },
        );

        Self {
            fname,
            input_idents,
            functype: Type::Func {
                inputs,
                span: None,
                output: out_type.into(),
            },
            body,
            span,
            doc_comment,
        }
    }

    pub fn airity(&self) -> u16 {
        self.input_idents.len() as u16
    }

    pub fn fntype(&self) -> &Type {
        &self.functype
    }

    pub fn name(&self) -> &String {
        &self.fname.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.fname.span()
    }

    pub fn body(&self) -> &Expression {
        &self.body
    }

    pub fn input_idents(&self) -> &Vec<Ident> {
        &self.input_idents
    }

    pub fn output_span(&self) -> Option<Span> {
        match &self.functype {
            Type::Func { output, .. } => match output.as_ref() {
                Some(a) => a.span(),
                None => None,
            },
            _ => None,
        }
    }
}
