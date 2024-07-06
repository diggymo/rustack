// fn recognizer(input: &str) -> &str;

use core::panic;
use std::convert::identity;
use std::io::Read;
use std::{error::Error, ops::Range, str::Chars, vec};

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

type Statements<'a> = Vec<Expression<'a>>;

fn statements(input: &str) -> Result<Statements, nom::error::Error<&str>> {
    let (_, res) = separated_list0(tag(";"), expr)(input).unwrap();
    Ok(res)
}

fn main() {
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
            println!("eval: {:?}", eval(statement));
        }
    }
}

fn eval(expr: Expression) -> f64 {
    match expr {
        Expression::Ident("pi") => std::f64::consts::PI,
        Expression::Ident(id) => panic!("Unknown name {:?}", id),
        Expression::NumLiteral(n) => n,
        Expression::Add(lhs, rhs) => eval(*lhs) + eval(*rhs),
        Expression::Sub(lhs, rhs) => eval(*lhs) - eval(*rhs),
        Expression::Mul(lhs, rhs) => eval(*lhs) * eval(*rhs),
        Expression::Div(lhs, rhs) => eval(*lhs) / eval(*rhs),
        Expression::FnInvoke("sqrt", args) => unary_fn(f64::sqrt)(args),
        Expression::FnInvoke("sin", args) => unary_fn(f64::sin)(args),
        Expression::FnInvoke("cos", args) => unary_fn(f64::cos)(args),
        Expression::FnInvoke("tan", args) => unary_fn(f64::tan)(args),
        Expression::FnInvoke("asin", args) => unary_fn(f64::asin)(args),
        Expression::FnInvoke("acos", args) => unary_fn(f64::acos)(args),
        Expression::FnInvoke("atan", args) => unary_fn(f64::atan)(args),
        Expression::FnInvoke("atan2", args) => binary_fn(f64::atan2)(args),
        Expression::FnInvoke("pow", args) => binary_fn(f64::powf)(args),
        Expression::FnInvoke("exp", args) => unary_fn(f64::exp)(args),
        Expression::FnInvoke("log", args) => binary_fn(f64::log)(args),
        Expression::FnInvoke("log10", args) => unary_fn(f64::log10)(args),
        Expression::FnInvoke(name, _) => {
            panic!("Unknown function {name:?}")
        }
    }
}

fn unary_fn(f: fn(f64) -> f64) -> impl Fn(Vec<Expression>) -> f64 {
    move |args| {
        f(eval(
            args.into_iter().next().expect("function missing argument"),
        ))
    }
}

fn binary_fn(f: fn(f64, f64) -> f64) -> impl Fn(Vec<Expression>) -> f64 {
    move |args| {
        let mut iter = args.into_iter();

        let lhs = eval(iter.next().unwrap());
        let rhs = eval(iter.next().unwrap());
        f(lhs, rhs)
    }
}

fn ex_eval<'src>(input: &'src str) -> Result<f64, nom::Err<nom::error::Error<&'src str>>> {
    expr(input).map(|(_, e)| eval(e))
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
        assert_eq!(ex_eval("123"), Ok(123.));
    }
    #[test]
    fn test_eval_2() {
        assert_eq!(ex_eval("(123 + 456) + pi"), Ok(582.1415926535898));
    }
    #[test]
    fn test_eval_3() {
        assert_eq!(ex_eval("10 + (100+1)"), Ok(111.));
    }
    #[test]
    fn test_eval_4() {
        assert_eq!(ex_eval("((1+2)+(3+4))+5+6"), Ok(21.));
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
        assert_eq!(ex_eval("2 * pi"), Ok(6.283185307179586));
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(ex_eval("(123 * 456 ) +pi)"), Ok(56091.14159265359));
    }

    #[test]
    fn test_eval_7() {
        assert_eq!(ex_eval("10 - ( 100 + 1 )"), Ok(-91.));
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(ex_eval("(3+7) /(2+3)"), Ok(2.));
    }

    #[test]
    fn test_eval_9() {
        assert_eq!(ex_eval("2 * 3 / 3"), Ok(2.));
    }

    #[test]
    fn test_fn_invoke_1() {
        assert_eq!(ex_eval("sqrt(2) / 2"), Ok(0.7071067811865476));
    }

    #[test]
    fn test_fn_invoke_2() {
        assert_eq!(ex_eval("sin(pi / 4)"), Ok(0.7071067811865475));
    }

    #[test]
    fn test_fn_invoke_3() {
        assert_eq!(ex_eval("atan2(1,1)"), Ok(0.7853981633974483));
    }
}
