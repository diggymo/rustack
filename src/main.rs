// fn recognizer(input: &str) -> &str;

use core::panic;
use std::{ops::Range, str::Chars, vec, error::Error};

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Ident,
    Number,
    LParen,
    RParen,
}

/**
 * ライフタイムは不要っぽい
 * `LParen`および`RParen`は含まれていない
 */
#[derive(Debug, PartialEq)]
enum TokenTree {
    Token(Token),
    Tree(Vec<TokenTree>),
}

#[derive(Debug, PartialEq)]
enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
}


fn eval(expr: Expression) -> f64 {
    match expr {
        Expression::Ident("pi") => std::f64::consts::PI,
        Expression::Ident(id) => panic!("Unknown name {:?}", id),
        Expression::NumLiteral(n) => n,
        Expression::Add(lhs, rhs) => eval(*lhs) + eval(*rhs)
    }
}

fn ex_eval<'src>(
    input: &'src str
) -> Option<f64> {
    expr(input).map(|(_,e)| eval(e))
}

/**
 * 式。項+加算。
 * `((-1))` `+5` `3+2`
 */
fn expr(input: &str) -> Option<(&str, Expression)> {
    // `3+2`
    if let Some(res) = add(input) {
        return Some(res);
    }

    // `((-1))` `+5`
    if let Some(res) = term(input) {
        return Some(res);
    }

    None
}

/** 不要な気がする... */
fn advance_char(input: &str) -> &str {
    let mut chars = input.chars();
    chars.next();
    chars.as_str()
}

fn peek_char(input: &str) -> Option<char> {
    input.chars().next()
}

/** 文字列を1つのかたまり（=token）に切り出す */
fn token(input: &str) -> Option<(&str, Expression)> {
    let trimed_input = whitespace(input);
    if let Some((i, ident_res)) = ident(trimed_input) {
        return Some((i, ident_res));
    }

    if let Some((i, number_res)) = number(trimed_input) {
        dbg!(&i, &number_res);
        return Some((i, number_res));
    }

    None
}

fn whitespace(mut input: &str) -> &str {
    // input.chars()は新しいiteratorを生み出す。
    let mut chars = input.chars();
    while matches!(chars.next(), Some(' ')) {
        input = chars.as_str();
    }

    input
}

/**
 * 加算
 * `1+2+3`
 */
fn add(input: &str) -> Option<(&str, Expression)> {
    // 最初の項をパースする
    // ("2+3", Numeric(1)) = add_term("1+2+3")
    let (mut remaining, mut result) = add_term(input)?;

    // 残りの項を処理する
    // ("3", Numeric(2)) = add_term("2+3")
    while let Some((next_input, term)) = add_term(remaining) {
        // Add(1, 2)
        result = Expression::Add(Box::new(result), Box::new(term));
        // "3"
        remaining = next_input;
    }

    // 式の残りをパースする
    // ("", Numeric(3)) = expr("3")
    let (final_input, rhs) = expr(remaining)?;

    // 最終的な加算式を構築する
    // Add(Add(1, 2), 3)
    let final_expression = Expression::Add(Box::new(result), Box::new(rhs));

    Some((final_input, final_expression))
}

fn add_term(mut input: &str) -> Option<(&str, Expression)> {
    let (next_input, lhs) = term(input)?;

    let next_input = plus(whitespace(next_input))?;

    Some((next_input, lhs))
}

/**
 * `+`であるか判定
 */
fn plus(input: &str) -> Option<&str> {
    if peek_char(input) != Some('+') {
        return None;
    }

    let mut chars = input.chars();
    chars.next();
    Some(chars.as_str())
}

/**
 * 項1つ分を評価するパーサー
 * ex: `(-1) + 5`→ "+ 5" と numericVal(-1)
 * ex: `1 + 5`→ "+ 5" と numericVal(1)
 */
fn term(input: &str) -> Option<(&str, Expression)> {
    dbg!("term", &input);
    if let Some(res) = paren(input) {
        return Some(res);
    }

    if let Some(res) = token(input) {
        return Some(res);
    }

    None
}

fn paren(input: &str) -> Option<(&str, Expression)> {
    let next_input = l_paren(whitespace(input))?;

    let (next_input, expr) = expr(next_input)?;

    let next_input = r_paren(whitespace(next_input))?;

    Some((next_input, expr))
}

fn number(mut input: &str) -> Option<(&str, Expression)> {
    let start = input;
    dbg!(start);
    let first_char = peek_char(input);
    if !matches!(first_char, Some(_x @ ('-' | '+' | '.' | '0'..='9'))) {
        return None;
    }

    let mut chars = input.chars();

    // NOTE: 初回のCharの検査はスキップする（`-`や`+`に対応できないため）
    let mut index = 1;
    chars.next();
    input = chars.as_str();

    while matches!(chars.next(), Some(_x @ ('.' | '0'..='9'))) {
        index += 1;
        input = chars.as_str();
    }

    dbg!(&input, &index);

    Some((
        input,
        Expression::NumLiteral(start[0..index].parse().unwrap()),
    ))
}

fn ident(mut input: &str) -> Option<(&str, Expression)> {
    let start = input;
    if !matches!(peek_char(input), Some(_x @ ('a'..='z' | 'A'..='Z'))) {
        return None;
    }

    let mut chars = input.chars();
    let mut index = 0;
    while matches!(chars.next(), Some(_x @ ('a'..='z' | 'A'..='Z' | '0'..='9'))) {
        index += 1;
        input = chars.as_str();
    }

    Some((input, Expression::Ident(&start[0..index])))
}

fn l_paren(input: &str) -> Option<&str> {
    if peek_char(input) == Some('(') {
        let mut chars = input.chars();
        chars.next();
        return Some(chars.as_str());
    }

    None
}

fn r_paren(input: &str) -> Option<&str> {
    if peek_char(input) == Some(')') {
        let mut chars = input.chars();
        chars.next();
        return Some(chars.as_str());
    }

    None
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_white_space() {
        let result = whitespace(" abc");
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_white_space_2() {
        let result = whitespace("abc");
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_number() {
        let result = number("12.34");
        assert_eq!(result, Some(("", Expression::NumLiteral(12.34))));
    }

    #[test]
    fn test_number_2() {
        let result = number("12.34  ");
        assert_eq!(result, Some(("  ", Expression::NumLiteral(12.34))));
    }

    #[test]
    fn test_ident() {
        let result = ident("hogehoge  ");
        assert_eq!(result, Some(("  ", Expression::Ident("hogehoge"))));
    }

    #[test]
    fn test_ident_2() {
        let result = ident("hoge");
        assert_eq!(result, Some(("", Expression::Ident("hoge"))));
    }

    #[test]
    fn test_term() {
        let result = term("(+5)");
        assert_eq!(result, Some(("", Expression::NumLiteral(5.0))));
    }

    #[test]
    fn test_term_minus() {
        let result = term("(-5)+1");
        assert_eq!(result, Some(("+1", Expression::NumLiteral(-5.0))));
    }

    #[test]
    fn test_term_plus() {
        let result = term("((1 + 2) + (3 + 4)) + 5 + 6");
        assert_eq!(
            result,
            Some((
                " + 5 + 6",
                Expression::Add(
                    Box::new(Expression::Add(
                        Box::new(Expression::NumLiteral(1.)),
                        Box::new(Expression::NumLiteral(2.)),
                    )),
                    Box::new(Expression::Add(
                        Box::new(Expression::NumLiteral(3.)),
                        Box::new(Expression::NumLiteral(4.)),
                    ))
                )
            ))
        );
    }


    #[test]
    fn test_eval_1() {
        assert_eq!(ex_eval("123"), Some(123.));
    }
    #[test]
    fn test_eval_2() {
        assert_eq!(ex_eval("(123 + 456) + pi"), Some(582.1415926535898));
    }
    #[test]
    fn test_eval_3() {
        assert_eq!(ex_eval("10 + (100+1)"), Some(111.));
    }
    #[test]
    fn test_eval_4() {
        assert_eq!(ex_eval("((1+2)+(3+4))+5+6"), Some(21.));
    }
}

fn main() {}
