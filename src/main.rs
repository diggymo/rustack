// fn recognizer(input: &str) -> &str;

use core::panic;
use std::convert::identity;
use std::{error::Error, ops::Range, str::Chars, vec};

use nom::bytes::complete::tag;
use nom::character::complete::{
    alpha1, alphanumeric0, alphanumeric1, char, digit0, digit1, one_of,
};
use nom::combinator::{opt, recognize};
use nom::multi::many0;
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
    dbg!("expr_start");
    let (i, init) = term(input)?;

    dbg!("expr", &init);

    fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val)| match op {
            '+' => Expression::Add(Box::new(acc), Box::new(val)),
            '-' => Expression::Sub(Box::new(acc), Box::new(val)),
            _ => panic!("aaa")
        },
    )(i)
}

fn term(input: &str) -> IResult<&str, Expression> {
    dbg!("term_start");
    let (i, init) = factor(input)?;

    dbg!("term", &init);

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
    alt((number, ident, paren))(i)
}
fn paren(input: &str) -> IResult<&str, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(input)
}

fn number(mut input: &str) -> IResult<&str, Expression> {
    space_delimited(recognize_float)(input)
        .map(|(i, a)| (i, Expression::NumLiteral(a.parse().unwrap())))
}

/** 自作したコンビネーターのシグネチャからnomのシグネチャに変換するwrapper */
fn option2result(
    f: impl Fn(&str) -> Option<(&str, Expression)>,
) -> impl FnMut(&str) -> Result<(&str, Expression), nom::Err<nom::error::Error<&str>>> {
    return move |i| {
        let result = f(i);
        result.ok_or(nom::Err::Incomplete(nom::Needed::Unknown))
    };
}

fn ident(mut input: &str) -> IResult<&str, Expression> {
    space_delimited(recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    )))(input)
    .map(|(next_input, a)| return (next_input, Expression::Ident(a)))
}

fn space_delimited<'src, O, E>(
    f: impl Parser<&'src str, O, E>,
) -> impl FnMut(&'src str) -> IResult<&'src str, O, E>
where
    E: ParseError<&'src str>,
{
    delimited(multispace0, f, multispace0)
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

    // #[test]
    // fn test_term() {
    //     let result = term("(+5)");
    //     assert_eq!(result, Ok(("", Expression::NumLiteral(5.0))));
    // }

    // #[test]
    // fn test_term_minus() {
    //     let result = term("(-5)+1");
    //     assert_eq!(result, Ok(("+1", Expression::NumLiteral(-5.0))));
    // }

    // #[test]
    // fn test_term_plus() {
    //     let result = term("((1 + 2) + (3 + 4)) + 5 + 6");
    //     assert_eq!(
    //         result,
    //         Ok((
    //             " + 5 + 6",
    //             Expression::Add(
    //                 Box::new(Expression::Add(
    //                     Box::new(Expression::NumLiteral(1.)),
    //                     Box::new(Expression::NumLiteral(2.)),
    //                 )),
    //                 Box::new(Expression::Add(
    //                     Box::new(Expression::NumLiteral(3.)),
    //                     Box::new(Expression::NumLiteral(4.)),
    //                 ))
    //             )
    //         ))
    //     );
    // }

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
}

fn main() {}
