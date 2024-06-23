// fn recognizer(input: &str) -> &str;

fn whitespace(mut input: &str) -> &str {
    // input.chars()は新しいiteratorを生み出す。
    let mut chars = input.chars();
    while matches!(chars.next(), Some(' ')) {
        input = chars.as_str();
    }

    input
}

fn number(mut input: &str) -> &str {
    let first_char = input.chars().next();
    if matches!(first_char, Some(_x @ ('-'|'+'|'.'|'0'..='9'))) {
        // let mut result: &str = input;
        // while matches!(input.chars().next(), Some(_x @ ('.'|'0'..='9'))) {
        //     let mut chars = input.chars();
        //     chars.next();
        //     input = chars.as_str();
        // }

        let mut chars = input.chars();
        while matches!(chars.next(), Some(_x @ ('.'|'0'..='9'))) {
            input = chars.as_str();
        }
    }
    input
}


fn ident(mut input: &str) -> &str {
    if matches!(input.chars().next(), Some(_x @ ('a'..='z'|'A'..='Z'))) {
        let mut chars = input.chars();
        while matches!(chars.next(), Some(_x @ ('a'..='z'|'A'..='Z'| '0'..='9'))) {
            input  =chars.as_str();
        }
    }
    input
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

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
        assert_eq!(result, "");
    }


    #[test]
    fn test_number_2() {
        let result = number("12.34  ");
        assert_eq!(result, "  ");
    }


    #[test]
    fn test_ident() {
        let result = ident("hogehoge  ");
        assert_eq!(result, "  ");
    }


    #[test]
    fn test_ident_2() {
        let result = ident("hoge");
        assert_eq!(result, "");
    }

}

fn main() {

}
// fn main() {
//     let mut a = "    vv ## ##";
//     let result = whitespace(&mut a);
//     dbg!(result, a);
    
// }