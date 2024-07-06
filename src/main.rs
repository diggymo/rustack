// fn recognizer(input: &str) -> &str;

use core::panic;
use std::collections::HashMap;
use std::convert::identity;
use std::io::Read;
use std::{error::Error, ops::Range, str::Chars, vec};

use nom::Err;
use nom::branch::permutation;
use nom::bytes::complete::tag;
use nom::character::complete::{
    alpha1, alphanumeric0, alphanumeric1, char, digit0, digit1, one_of,
};
use nom::combinator::{opt, recognize};
use nom::multi::{many0, separated_list0};
use nom::number::complete::recognize_float;
use nom::{
    branch::alt,
    character::complete::multispace0,
    error::ParseError,
    multi::fold_many0,
    sequence::{delimited, pair},
    IResult, Parser,
};

#[derive(Debug, PartialEq, Clone)]
enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    FnInvoke(&'src str, Vec<Expression<'src>>),
}

type Statements<'a> = Vec<Statement<'a>>;

enum Statement<'src> {
    // 式文
    // `get_value();` や `register_user(name);`など
    Expression(Expression<'src>),
    // 変数宣言
    VarDef(&'src str, Expression<'src>)
}

fn statements(input: &str) -> Result<Statements, nom::error::Error<&str>> {
    let (_, res) = separated_list0(tag(";"), statement)(input).unwrap();
    Ok(res)
}

fn statement(input: &str) -> IResult<&str, Statement> {
   space_delimited(alt((var_def, expr_statement)))(input)
}

fn var_def(input: &str) -> IResult<&str, Statement> {
    permutation((
        space_delimited(tag("var")), 
        space_delimited(identifier), 
        space_delimited(char('=')), 
        space_delimited(expr)
    ))(input).map(|(next_input,parsed) | {
        (next_input, Statement::VarDef(parsed.1, parsed.3))
    })
}
fn expr_statement(input: &str) -> IResult<&str, Statement> {
    expr(input).map(|(next_input, parsed_expression)| {
        (next_input, Statement::Expression(parsed_expression))
    })
}

fn main() {

    let mut variables = HashMap::new();

    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_ok() {
        let parsed_statements = match statements(&buf) {
            Ok(parsed_statements) => parsed_statements,
            Err(e) => {
                eprintln!("parsed error: {e:?}");
                return;
            }
        };

        for statement in parsed_statements {
            match statement {
                Statement::VarDef(identifier,expression ) => {
                    variables.insert(identifier, eval(expression, &variables));
                },
                Statement::Expression(expression) => {
                    println!("eval: {:?}", eval(expression, &variables));
                }
            }
        }
    }
}

fn eval(expr: Expression, vars: &HashMap<&str, f64>) -> f64 {
    match expr {
        Expression::Ident("pi") => std::f64::consts::PI,
        Expression::Ident(id) => *vars.get(id).expect("Unknown name {:?}"),
        Expression::NumLiteral(n) => n,
        Expression::Add(lhs, rhs) => eval(*lhs, vars) + eval(*rhs, vars),
        Expression::Sub(lhs, rhs) => eval(*lhs, vars) - eval(*rhs, vars),
        Expression::Mul(lhs, rhs) => eval(*lhs, vars) * eval(*rhs, vars),
        Expression::Div(lhs, rhs) => eval(*lhs, vars) / eval(*rhs, vars),
        Expression::FnInvoke("sqrt", args) => unary_fn(f64::sqrt)(args, vars),
        Expression::FnInvoke("sin", args) => unary_fn(f64::sin)(args, vars),
        Expression::FnInvoke("cos", args) => unary_fn(f64::cos)(args, vars),
        Expression::FnInvoke("tan", args) => unary_fn(f64::tan)(args, vars),
        Expression::FnInvoke("asin", args) => unary_fn(f64::asin)(args, vars),
        Expression::FnInvoke("acos", args) => unary_fn(f64::acos)(args, vars),
        Expression::FnInvoke("atan", args) => unary_fn(f64::atan)(args, vars),
        Expression::FnInvoke("atan2", args) => binary_fn(f64::atan2)(args, vars),
        Expression::FnInvoke("pow", args) => binary_fn(f64::powf)(args, vars),
        Expression::FnInvoke("exp", args) => unary_fn(f64::exp)(args, vars),
        Expression::FnInvoke("log", args) => binary_fn(f64::log)(args, vars),
        Expression::FnInvoke("log10", args) => unary_fn(f64::log10)(args, vars),
        Expression::FnInvoke(name, _) => {
            panic!("Unknown function {name:?}")
        }
    }
}

fn unary_fn(f: fn(f64) -> f64) -> impl Fn(Vec<Expression>, &HashMap<&str, f64>) -> f64 {
    move |args, variables| {
        f(eval(
            args.into_iter().next().expect("function missing argument"),
            variables
        ))
    }
}

fn binary_fn(f: fn(f64, f64) -> f64) -> impl Fn(Vec<Expression>, &HashMap<&str, f64>) -> f64 {
    move |args, variables| {
        let mut iter = args.into_iter();

        let lhs = eval(iter.next().unwrap(), variables);
        let rhs = eval(iter.next().unwrap(), variables);
        f(lhs, rhs)
    }
}

fn ex_eval<'src>(input: &'src str, vars: &HashMap<&str, f64>) -> Result<f64, nom::Err<nom::error::Error<&'src str>>> {
    expr(input).map(|(_, e)| eval(e, vars))
}

/**
 * 式。項+加算。
 * `((-1))` `+5` `3+2`
 */
fn expr(input: &str) -> IResult<&str, Expression> {
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
        assert_eq!(ex_eval("123", &HashMap::new()), Ok(123.));
    }
    #[test]
    fn test_eval_2() {
        assert_eq!(ex_eval("(123 + 456) + pi", &HashMap::new()), Ok(582.1415926535898));
    }
    #[test]
    fn test_eval_3() {
        assert_eq!(ex_eval("10 + (100+1)", &HashMap::new()), Ok(111.));
    }
    #[test]
    fn test_eval_4() {
        assert_eq!(ex_eval("((1+2)+(3+4))+5+6", &HashMap::new()), Ok(21.));
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
        assert_eq!(ex_eval("2 * pi", &HashMap::new()), Ok(6.283185307179586));
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(ex_eval("(123 * 456 ) +pi)", &HashMap::new()), Ok(56091.14159265359));
    }

    #[test]
    fn test_eval_7() {
        assert_eq!(ex_eval("10 - ( 100 + 1 )", &HashMap::new()), Ok(-91.));
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(ex_eval("(3+7) /(2+3)", &HashMap::new()), Ok(2.));
    }

    #[test]
    fn test_eval_9() {
        assert_eq!(ex_eval("2 * 3 / 3", &HashMap::new()), Ok(2.));
    }

    #[test]
    fn test_fn_invoke_1() {
        assert_eq!(ex_eval("sqrt(2) / 2", &HashMap::new()), Ok(0.7071067811865476));
    }

    #[test]
    fn test_fn_invoke_2() {
        assert_eq!(ex_eval("sin(pi / 4)", &HashMap::new()), Ok(0.7071067811865475));
    }

    #[test]
    fn test_fn_invoke_3() {
        assert_eq!(ex_eval("atan2(1,1)", &HashMap::new()), Ok(0.7853981633974483));
    }
}
