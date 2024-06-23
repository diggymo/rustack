// fn recognizer(input: &str) -> &str;

use std::{ops::Range, str::Chars};

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Ident,
    Number,
    LParen,
    RParen,
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

fn source(mut input: &str) -> Vec<Token> {
    let mut result = vec![];
    loop {
        if let (new_input, Some(token_type)) = token(input) {
            result.push(token_type);
            input = new_input;
        } else {
            break;
        }
    }

    result
}

fn token(input: &str) -> (&str, Option<Token>) {
    let trimed_input = whitespace(input);
    if let (i, Some(ident_res)) = ident(trimed_input) {
        return (i, Some(ident_res));
    }

    if let (i, Some(number_res)) = number(trimed_input) {
        return (i, Some(number_res));
    }

    if let (i, Some(paren_res)) = paren(trimed_input) {
        return (i, Some(paren_res));
    }

    (input, None)
}

fn whitespace(mut input: &str) -> &str {
    // input.chars()は新しいiteratorを生み出す。
    let mut chars = input.chars();
    while matches!(chars.next(), Some(' ')) {
        input = chars.as_str();
    }

    input
}

fn number(mut input: &str) -> (&str, Option<Token>) {
    let first_char = peek_char(input);
    if !matches!(first_char, Some(_x @ ('-' | '+' | '.' | '0'..='9'))) {
        return (input, None);
    }

    let mut chars = input.chars();
    while matches!(chars.next(), Some(_x @ ('.' | '0'..='9'))) {
        input = chars.as_str();
    }
    (input, Some(Token::Number))
}

fn ident(mut input: &str) -> (&str, Option<Token>) {
    if !matches!(peek_char(input), Some(_x @ ('a'..='z' | 'A'..='Z'))) {
        return (input, None);
    }
    let mut chars = input.chars();
    while matches!(chars.next(), Some(_x @ ('a'..='z' | 'A'..='Z' | '0'..='9'))) {
        input = chars.as_str();
    }
    (input, Some(Token::Ident))
}

fn paren(input: &str) -> (&str, Option<Token>) {
    if peek_char(input) == Some('(') {
        let mut chars = input.chars();
        chars.next();
        return (chars.as_str(), Some(Token::LParen));
    }

    if peek_char(input) == Some(')') {
        let mut chars = input.chars();
        chars.next();
        return (chars.as_str(), Some(Token::RParen));
    }

    (input, None)
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
        assert_eq!(result.0, "");
    }

    #[test]
    fn test_number_2() {
        let result = number("12.34  ");
        assert_eq!(result.0, "  ");
    }

    #[test]
    fn test_ident() {
        let result = ident("hogehoge  ");
        assert_eq!(result.0, "  ");
    }

    #[test]
    fn test_ident_2() {
        let result = ident("hoge");
        assert_eq!(result.0, "");
    }

    #[test]
    fn test_source() {
        let result = source("123 world");
        assert_eq!(result, vec![Token::Number, Token::Ident]);
    }

    #[test]
    fn test_source_2() {
        let result = source("Hello world");
        assert_eq!(result, vec![Token::Ident, Token::Ident]);
    }

    #[test]
    fn test_source_3() {
        let result = source("      world");
        assert_eq!(result, vec![Token::Ident]);
    }

    #[test]
    fn test_source_empty() {
        let result = source("");
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_source_paren() {
        let result = source("() (())");
        assert_eq!(
            result,
            vec![
                Token::LParen,
                Token::RParen,
                Token::LParen,
                Token::LParen,
                Token::RParen,
                Token::RParen,
            ]
        );
    }
}

fn main() {}
