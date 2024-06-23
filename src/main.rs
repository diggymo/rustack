// fn recognizer(input: &str) -> &str;

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Ident,
    Number,
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
    let first_char = input.chars().next();
    if !matches!(first_char, Some(_x @ ('-' | '+' | '.' | '0'..='9'))) {
        return (input, None);
    }
    // let mut result: &str = input;
    // while matches!(input.chars().next(), Some(_x @ ('.'|'0'..='9'))) {
    //     let mut chars = input.chars();
    //     chars.next();
    //     input = chars.as_str();
    // }

    let mut chars = input.chars();
    while matches!(chars.next(), Some(_x @ ('.' | '0'..='9'))) {
        input = chars.as_str();
    }
    (input, Some(Token::Number))
}

fn ident(mut input: &str) -> (&str, Option<Token>) {
    if !matches!(input.chars().next(), Some(_x @ ('a'..='z' | 'A'..='Z'))) {
        return (input, None);
    }
    let mut chars = input.chars();
    while matches!(chars.next(), Some(_x @ ('a'..='z' | 'A'..='Z' | '0'..='9'))) {
        input = chars.as_str();
    }
    (input, Some(Token::Ident))
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
}

fn main() {}

