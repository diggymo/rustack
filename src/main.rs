// fn recognizer(input: &str) -> &str;

use core::panic;
use std::collections::HashMap;
use std::io::Read;

use nom::branch::permutation;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, char, space0, space1};
use nom::combinator::{opt, recognize};
use nom::multi::{many0, separated_list0};
use nom::number::complete::recognize_float;
use nom::sequence::terminated;
use nom::{
    branch::alt,
    character::complete::multispace0,
    error::ParseError,
    multi::fold_many0,
    sequence::{delimited, pair},
    IResult, Parser,
};

type Variables = HashMap<String, f64>;
type Functions<'src> = HashMap<String, FnDef<'src>>;

struct StackFrame<'src> {
    vars: Variables,
    funcs: Functions<'src>,
    uplevel: Option<&'src StackFrame<'src>>,
}

impl<'src> StackFrame<'src> {
    pub fn push_stack(uplevel: &'src Self) -> StackFrame<'src> {
        Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            uplevel: Some(uplevel),
        }
    }

    pub fn new() -> StackFrame<'src> {
        let mut funcs = Functions::new();
        funcs.insert("sqrt".to_string(), unary_fn(f64::sqrt));
        funcs.insert("sin".to_string(), unary_fn(f64::sin));
        funcs.insert("cos".to_string(), unary_fn(f64::cos));
        funcs.insert("tan".to_string(), unary_fn(f64::tan));
        funcs.insert("asin".to_string(), unary_fn(f64::asin));
        funcs.insert("acos".to_string(), unary_fn(f64::acos));
        funcs.insert("atan".to_string(), unary_fn(f64::atan));
        funcs.insert("atan2".to_string(), binary_fn(f64::atan2));
        funcs.insert("pow".to_string(), binary_fn(f64::powf));
        funcs.insert("exp".to_string(), unary_fn(f64::exp));
        funcs.insert("log".to_string(), binary_fn(f64::log));
        funcs.insert("log10".to_string(), unary_fn(f64::log10));
        funcs.insert("print".to_string(), unary_fn(_print));
        StackFrame {
            vars: HashMap::new(),
            funcs: funcs,
            uplevel: None,
        }
    }

    pub fn get_fn(&self, name: &str) -> Option<&FnDef> {
        if let Some(fn_def) = self.funcs.get(name) {
            return Some(fn_def);
        }

        if let Some(frame) = self.uplevel {
            return frame.get_fn(name);
        }

        return None;
    }

    pub fn get_var(&self, name: &str) -> &f64 {
        if let Some(var_value) = self.vars.get(name) {
            return var_value;
        }

        if let Some(frame) = self.uplevel {
            return frame.get_var(name);
        }

        panic!("存在しない変数が参照されました name={}", name);
    }

    pub fn set_var(&mut self, name: &str, value: f64) -> &f64 {
        if let Some(_) = self.vars.get(name) {
            self.vars.insert(name.to_string(), value);
        }

        // if let Some(frame) = self.uplevel {
        //     // frame.set_var(name, value);
        // }
        // NOTE: 親スタックの内容を改変しないといけないため、難しくなる
        panic!(
            "親スタックの値は変更できません...もしくは存在しない変数が参照されました name={}",
            name
        );
    }
}

fn _print(a: f64) -> f64 {
    println!("{a}");
    a
}

#[derive(Debug, PartialEq, Clone)]
enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    FnInvoke(&'src str, Vec<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
}

type Statements<'a> = Vec<Statement<'a>>;

#[derive(Debug, PartialEq, Clone)]
enum Statement<'src> {
    // 式文
    // `get_value();` や `register_user(name);`など
    Expression(Expression<'src>),
    // 変数宣言
    VarDef(&'src str, Expression<'src>),
    // 変数代入
    VarAssign(&'src str, Expression<'src>),

    For {
        loop_var: &'src str,
        start: Expression<'src>,
        end: Expression<'src>,
        stmts: Statements<'src>,
    },

    FnDef {
        name: &'src str,
        args: Vec<&'src str>,
        stms: Statements<'src>,
    },
    Return(Expression<'src>),
}

enum FnDef<'src> {
    User(UserFn<'src>),
    Native(NativeFn),
}

impl<'src> FnDef<'src> {
    pub fn call(&self, called_args: &[f64], frame: &StackFrame) -> f64 {
        match self {
            Self::User(UserFn { args, stmts }) => {
                let mut new_stack_frame = StackFrame::push_stack(frame);
                new_stack_frame.vars = called_args
                    .iter()
                    .zip(args.iter())
                    .map(|(arg, name)| (name.to_string(), *arg))
                    .collect();
                eval_stmts(stmts, &mut new_stack_frame)
            }
            Self::Native(NativeFn { code }) => code(called_args),
        }
    }
}

struct UserFn<'src> {
    args: Vec<&'src str>,
    stmts: Statements<'src>,
}
struct NativeFn {
    code: Box<dyn Fn(&[f64]) -> f64>,
}

fn statements(input: &str) -> IResult<&str, Statements> {
    let (input, stmts) = many0(statement)(input)?;
    // NOTE: 必要？
    // let (input, _) = opt(char(';'))(input)?;
    Ok((input, stmts))
}

fn statement(input: &str) -> IResult<&str, Statement> {
    alt((
        for_statement,
        fn_def_statement,
        return_statement,
        terminated(alt((var_def, var_assign, expr_statement)), char(';')),
    ))(input)
}

fn return_statement(input: &str) -> IResult<&str, Statement> {
    let (next_input, (_, expression, _)) = permutation((
        space_delimited(tag("return")),
        expr,
        space_delimited(char(';')),
    ))(input)?;

    return Ok((next_input, Statement::Return(expression)));
}

fn for_statement(input: &str) -> IResult<&str, Statement> {
    let (input, _) = space_delimited(tag("for"))(input)?;
    println!("#");
    let (input, (_, loop_identifier, _, _, start_expr, _, end_expr)) =
        permutation((space0, identifier, space1, tag("in"), expr, tag("to"), expr))(input)?;
    println!("##");

    let (input, statement_vec) = delimited(
        space_delimited(char('{')),
        many0(statement),
        space_delimited(char('}')),
    )(input)?;

    return Ok((
        input,
        Statement::For {
            loop_var: loop_identifier,
            start: start_expr,
            end: end_expr,
            stmts: statement_vec,
        },
    ));
}

fn fn_def_statement(input: &str) -> IResult<&str, Statement> {
    let (input, _) = space_delimited(tag("fn"))(input)?;

    let (input, (fn_name, args)) = permutation((
        space_delimited(identifier),
        delimited(
            char('('),
            separated_list0(char(','), space_delimited(identifier)),
            char(')'),
        ),
    ))(input)?;

    let (input, statements) = delimited(
        space_delimited(char('{')),
        many0(statement),
        space_delimited(char('}')),
    )(input)?;

    Ok((
        input,
        Statement::FnDef {
            name: fn_name,
            args: args,
            stms: statements,
        },
    ))
}

fn var_def(input: &str) -> IResult<&str, Statement> {
    permutation((
        space_delimited(tag("var")),
        space_delimited(identifier),
        space_delimited(char('=')),
        space_delimited(expr),
    ))(input)
    .map(|(next_input, parsed)| (next_input, Statement::VarDef(parsed.1, parsed.3)))
}

fn var_assign(input: &str) -> IResult<&str, Statement> {
    permutation((
        space_delimited(identifier),
        space_delimited(char('=')),
        space_delimited(expr),
    ))(input)
    .map(|(next_input, parsed)| (next_input, Statement::VarAssign(parsed.0, parsed.2)))
}

fn expr_statement(input: &str) -> IResult<&str, Statement> {
    expr(input).map(|(next_input, parsed_expression)| {
        (next_input, Statement::Expression(parsed_expression))
    })
}

fn main() {
    let mut buf = String::new();

    let mut stack_frame = StackFrame::new();
    if std::io::stdin().read_to_string(&mut buf).is_ok() {
        let parsed_statements = match statements(&buf) {
            Ok((_, parsed_statements)) => parsed_statements,
            Err(e) => {
                eprintln!("parsed error: {e:?}");
                return;
            }
        };
        dbg!(&parsed_statements);
        eval_stmts(&parsed_statements, &mut stack_frame);
    }
}

fn eval_stmts<'src>(stmts: &Statements<'src>, stack_frame: &mut StackFrame<'src>) -> f64 {
    let mut last_value: f64 = 0.;
    'statement_loop: for statement in stmts {
        match statement {
            Statement::VarDef(identifier, expression) => {
                let value = eval(expression, stack_frame);
                stack_frame.vars.insert(identifier.to_string(), value);
            }
            Statement::Return(return_expression) => {
                last_value = eval(return_expression, stack_frame);
                dbg!("##", &stmts, &return_expression, &last_value);
                // FIXME: eval_stmtsからbreakするだけでは不足している。さらに親のstmtsでbreakしなければならない
                break 'statement_loop;
            }
            Statement::VarAssign(identifier, expression) => {
                if !stack_frame.vars.contains_key(*identifier) {
                    panic!("存在していません");
                }
                let value = eval(expression, stack_frame);
                stack_frame
                    .vars
                    .insert(identifier.to_string(), value)
                    .unwrap();
            }
            Statement::For {
                loop_var,
                start,
                end,
                stmts: loop_stmts,
            } => {
                let start_value = eval(start, stack_frame);
                let end_value = eval(end, stack_frame);

                let mut i = start_value;
                while i < end_value {
                    stack_frame.vars.insert(loop_var.to_string(), i);
                    eval_stmts(loop_stmts, stack_frame);
                    i += 1.;
                }
            }
            Statement::Expression(expression) => {
                last_value = eval(expression, stack_frame);
            }
            Statement::FnDef { name, args, stms } => {
                stack_frame.funcs.insert(
                    name.to_string(),
                    FnDef::User(UserFn {
                        args: args.clone(),
                        stmts: stms.clone(),
                    }),
                );
            }
        }
    }

    dbg!(&last_value);
    last_value
}

fn eval<'src>(expr: &Expression<'src>, frame: &mut StackFrame<'src>) -> f64 {
    match expr {
        Expression::Ident("pi") => std::f64::consts::PI,
        Expression::Ident(id) => *frame.get_var(*id),
        Expression::NumLiteral(n) => *n,
        Expression::Add(lhs, rhs) => eval(lhs, frame) + eval(rhs, frame),
        Expression::Sub(lhs, rhs) => eval(lhs, frame) - eval(rhs, frame),
        Expression::Mul(lhs, rhs) => eval(lhs, frame) * eval(rhs, frame),
        Expression::Div(lhs, rhs) => eval(lhs, frame) / eval(rhs, frame),
        Expression::FnInvoke(name, args) => {
            let args: Vec<_> = args.into_iter().map(|e| eval(e, frame)).collect();
            if let Some(func) = frame.get_fn(*name) {
                func.call(&args, frame)
            } else {
                panic!("aaa");
            }
        }
        Expression::If(condition_ex, true_ex, false_ex) => {
            let result = eval(condition_ex, frame);
            if result != 0. {
                println!("##true##");
                eval_stmts(true_ex, frame)
            } else {
                println!("##false##");
                false_ex.as_ref().map_or(0., |_false_ex| {
                    return eval_stmts(_false_ex, frame);
                })
            }
        }
    }
}

fn unary_fn<'src>(f: fn(f64) -> f64) -> FnDef<'src> {
    FnDef::Native(NativeFn {
        code: Box::new(move |args| f(*args.into_iter().next().expect("function missing arg"))),
    })
}

fn binary_fn<'src>(f: fn(f64, f64) -> f64) -> FnDef<'src> {
    FnDef::Native(NativeFn {
        code: Box::new(move |args| {
            let mut iter = args.into_iter();
            f(
                *iter.next().expect("cant get first arg"),
                *iter.next().expect("cant get seconf arg"),
            )
        }),
    })
}

fn ex_eval<'src>(
    input: &'src str,
    frame: &mut StackFrame<'src>,
) -> Result<f64, nom::Err<nom::error::Error<&'src str>>> {
    expr(input).map(|(_, e)| eval(&e, frame))
}

/**
 * 式。項+加算。
 * `((-1))` `+5` `3+2`
 */
fn num_expr(input: &str) -> IResult<&str, Expression> {
    let (i, init) = term(input)?;

    fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val)| match op {
            '+' => Expression::Add(Box::new(acc), Box::new(val)),
            '-' => Expression::Sub(Box::new(acc), Box::new(val)),
            _ => panic!("aaa"),
        },
    )(i)
}

fn expr(input: &str) -> IResult<&str, Expression> {
    alt((if_expr, num_expr))(input)
}

fn if_expr(input: &str) -> IResult<&str, Expression> {
    let (next_input, _) = space_delimited(tag("if"))(input)?;
    permutation((
        space_delimited(expr),
        space_delimited(delimited(char('{'), space_delimited(statements), char('}'))),
        opt(permutation((
            space_delimited(tag("else")),
            space_delimited(delimited(char('{'), space_delimited(statements), char('}'))),
        ))),
    ))(next_input)
    .map(
        |(next_input, (condition, true_expression, false_expresion_option))| {
            return (
                next_input,
                Expression::If(
                    Box::new(condition),
                    Box::new(true_expression),
                    false_expresion_option.map(|(_, false_expresion)| Box::new(false_expresion)),
                ),
            );
        },
    )
}

fn term(input: &str) -> IResult<&str, Expression> {
    let (i, init) = factor(input)?;

    fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), factor),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| match op {
            '*' => Expression::Mul(Box::new(acc), Box::new(val)),
            '/' => Expression::Div(Box::new(acc), Box::new(val)),
            _ => panic!("Multiplicative expression should have '*' or '/' operator"),
        },
    )(i)
}

fn factor(i: &str) -> IResult<&str, Expression> {
    alt((func_call, number, ident, paren))(i)
}
fn paren(input: &str) -> IResult<&str, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(input)
}

fn number(mut input: &str) -> IResult<&str, Expression> {
    space_delimited(recognize_float)(input)
        .map(|(i, a)| (i, Expression::NumLiteral(a.parse().unwrap())))
}

fn ident(mut input: &str) -> IResult<&str, Expression> {
    space_delimited(identifier)(input)
        .map(|(next_input, a)| return (next_input, Expression::Ident(a)))
}

fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn space_delimited<'src, O, E>(
    f: impl Parser<&'src str, O, E>,
) -> impl FnMut(&'src str) -> IResult<&'src str, O, E>
where
    E: ParseError<&'src str>,
{
    delimited(multispace0, f, multispace0)
}

fn func_call(input: &str) -> IResult<&str, Expression> {
    let (input, ident_expression) = space_delimited(identifier)(input)?;

    let (input, args) = delimited(
        tag("("),
        many0(delimited(multispace0, expr, space_delimited(opt(tag(","))))),
        tag(")"),
    )(input)?;

    return Ok((input, Expression::FnInvoke(ident_expression, args)));
}

#[cfg(test)]
mod test {
    use std::hash::Hash;

    use super::*;

    #[test]
    fn test_number() {
        let result = number("12.34");
        assert_eq!(result, Ok(("", Expression::NumLiteral(12.34))));
    }

    #[test]
    fn test_number_2() {
        let result = number("12.34  ");
        assert_eq!(result, Ok(("", Expression::NumLiteral(12.34))));
    }

    #[test]
    fn test_number_minus() {
        let result = number("  -12.34  ");
        assert_eq!(result, Ok(("", Expression::NumLiteral(-12.34))));
    }

    #[test]
    fn test_ident() {
        let result = ident("hogehoge  ");
        assert_eq!(result, Ok(("", Expression::Ident("hogehoge"))));
    }

    #[test]
    fn test_ident_2() {
        let result = ident("_h_oge");
        assert_eq!(result, Ok(("", Expression::Ident("_h_oge"))));
    }

    #[test]
    fn test_eval_1() {
        assert_eq!(ex_eval("123", &mut StackFrame::new()), Ok(123.));
    }
    #[test]
    fn test_eval_2() {
        assert_eq!(
            ex_eval("(123 + 456) + pi", &mut StackFrame::new()),
            Ok(582.1415926535898)
        );
    }
    #[test]
    fn test_eval_3() {
        assert_eq!(ex_eval("10 + (100+1)", &mut StackFrame::new()), Ok(111.));
    }
    #[test]
    fn test_eval_4() {
        assert_eq!(
            ex_eval("((1+2)+(3+4))+5+6", &mut StackFrame::new()),
            Ok(21.)
        );
    }

    #[test]
    fn test_term_mul() {
        assert_eq!(
            term("2*3"),
            Ok((
                "",
                Expression::Mul(
                    Box::new(Expression::NumLiteral(2.)),
                    Box::new(Expression::NumLiteral(3.))
                )
            ))
        );
    }

    #[test]
    fn test_eval_5() {
        assert_eq!(
            ex_eval("2 * pi", &mut StackFrame::new()),
            Ok(6.283185307179586)
        );
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(
            ex_eval("(123 * 456 ) +pi)", &mut StackFrame::new()),
            Ok(56091.14159265359)
        );
    }

    #[test]
    fn test_eval_7() {
        assert_eq!(
            ex_eval("10 - ( 100 + 1 )", &mut StackFrame::new()),
            Ok(-91.)
        );
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(ex_eval("(3+7) /(2+3)", &mut StackFrame::new()), Ok(2.));
    }

    #[test]
    fn test_eval_9() {
        assert_eq!(ex_eval("2 * 3 / 3", &mut StackFrame::new()), Ok(2.));
    }

    #[test]
    fn test_fn_invoke_1() {
        assert_eq!(
            ex_eval("sqrt(2) / 2", &mut StackFrame::new()),
            Ok(0.7071067811865476)
        );
    }

    #[test]
    fn test_fn_invoke_2() {
        assert_eq!(
            ex_eval("sin(pi / 4)", &mut StackFrame::new()),
            Ok(0.7071067811865475)
        );
    }

    #[test]
    fn test_fn_invoke_3() {
        assert_eq!(
            ex_eval("atan2(1,1)", &mut StackFrame::new()),
            Ok(0.7853981633974483)
        );
    }

    #[test]
    fn test_delimited_with_multi_parentness() {
        assert_eq!(
            delimited(char('{'), if_expr, char('}'))("{if true {1;} else{2;} }")
                .unwrap()
                .1,
            Expression::If(
                Box::new(Expression::Ident("true")),
                Box::new(vec![Statement::Expression(Expression::NumLiteral(1.0))]),
                Some(Box::new(vec![Statement::Expression(
                    Expression::NumLiteral(2.0)
                )]))
            )
        )
    }

    #[test]
    fn test_for() {
        let result = for_statement("for i in 0 to 10 {i;}").unwrap();
        assert_eq!(
            result,
            (
                "",
                Statement::For {
                    loop_var: "i",
                    start: Expression::NumLiteral(0.),
                    end: Expression::NumLiteral(10.),
                    stmts: vec![Statement::Expression(Expression::Ident("i"))]
                }
            )
        );
    }

    #[test]
    fn test_if_true() {
        assert_eq!(
            eval(&expr("if 1 {5;}").unwrap().1, &mut StackFrame::new()),
            5.
        );
    }

    #[test]
    fn test_if_false() {
        assert_eq!(
            eval(&expr("if 0 {5;}").unwrap().1, &mut StackFrame::new()),
            0.
        );
    }

    #[test]
    fn test_if2() {
        assert_eq!(
            eval_stmts(
                &statements("var a = 1; if 1 {a=2;}else{a=3;}; a;")
                    .unwrap()
                    .1,
                &mut StackFrame::new()
            ),
            2.
        );
    }

    #[test]
    fn test_if3() {
        assert_eq!(
            eval_stmts(
                &statements("var a = 4; if 1 { print(a); } ;print(a);")
                    .unwrap()
                    .1,
                &mut StackFrame::new()
            ),
            4.
        );
    }

    #[test]
    fn test_if4() {
        assert_eq!(
            eval_stmts(
                &statements("var a = 4; var z = if 0 { print(a); } else {5;};print(z);")
                    .unwrap()
                    .1,
                &mut StackFrame::new()
            ),
            5.
        );
    }

    #[test]
    fn test_if5() {
        assert_eq!(
            eval_stmts(
                &statements("var a = 4; var z =if 0 {print(a);}else{5;};print(z);")
                    .unwrap()
                    .1,
                &mut StackFrame::new()
            ),
            5.
        );
    }

    #[test]
    fn test_if6() {
        assert_eq!(
            eval_stmts(
                &statements("var a = 1;var hoge = if 1 {print(a+1);};print(a);")
                    .unwrap()
                    .1,
                &mut StackFrame::new()
            ),
            1.
        );
    }
}
