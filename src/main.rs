use std::error::Error;

#[derive(Debug, PartialEq, Eq)]
enum Value<'src> {
    Num(i32),
    Op(&'src str),
    Block(Vec<Value<'src>>)
}


impl<'src> Value<'src> {
    fn as_num(&self) -> i32 {
        match self {
            Self::Num(i) => *i,
            _ => panic!("value is not a number...")
        }
    }
}

fn main() {
    for line in std::io::stdin().lines() {
        if line.is_err() {
            panic!("what happened!");
        }
        parse(&line.unwrap());
    }
}

fn parse<'a>(line: &'a str) -> Vec<Value<'a>>{
    let mut stack: Vec<Value> = vec!();
    let mut words: Vec<_> = line.split(" ").collect();

    while let Some((&word, mut rest)) = words.split_first() {
        if word == "{" {
            let value;
            (value, rest) = parse_block(rest);
            stack.push(value);
        } else if let Ok(num) = word.parse::<i32>() {
            stack.push(Value::Num(num));
        } else {
            match word {
                "+" => add(&mut stack),
                "-" => sub(&mut stack),
                "*" => mul(&mut stack),
                "/" => div(&mut stack),
                _ => panic!("{word} is aaaa")
            }
        }

        words = rest.to_vec();
    }

    return stack;
}



fn parse_block<'src, 'a>(input: &'a [&'src str]) -> (Value<'src>, &'a [&'src str]) {
    let mut tokens = vec![];
    let mut words = input;
    
    while let Some((&word, mut rest)) = words.split_first() {
        if word.is_empty() {
            break;
        }

        if word == "{" {
            let value;
            (value, rest) = parse_block(rest);
            tokens.push(value);
        } else if word == "}" {
            return (Value::Block(tokens), rest);
        } else if let Ok(num) = word.parse::<i32>() {
            tokens.push(Value::Num(num));
        } else {
            tokens.push(Value::Op(word));
        }

        words = rest;
    }

    (Value::Block(tokens), words)
}

fn add(stack: &mut Vec<Value>) {
    let left_hand = stack.pop().unwrap().as_num();
    let right_hand = stack.pop().unwrap().as_num();
    stack.push(Value::Num(left_hand+right_hand));
}

fn sub(stack: &mut Vec<Value>) {
    let right_hand = stack.pop().unwrap().as_num();
    let left_hand = stack.pop().unwrap().as_num();
    stack.push(Value::Num(left_hand-right_hand));
}

fn mul(stack: &mut Vec<Value>) {
    let left_hand = stack.pop().unwrap().as_num();
    let right_hand = stack.pop().unwrap().as_num();
    stack.push(Value::Num(right_hand*left_hand));
}

fn div(stack: &mut Vec<Value>) {
    let right_hand = stack.pop().unwrap().as_num();
    let left_hand = stack.pop().unwrap().as_num();
    stack.push(Value::Num(left_hand/right_hand));
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_group() {
        assert_eq!(
            parse("1 2 + { 3 4 }"), 
            vec![Value::Num(3), Value::Block(vec![Value::Num(3), Value::Num(4)])]
        );
    }
}