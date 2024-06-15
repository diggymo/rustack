use std::error::Error;

#[derive(Debug, PartialEq, Eq, Clone)]
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

    fn to_block(self) -> Vec<Value<'src>> {
        match self {
            Self::Block(val) => val,
            _ => panic!("###")
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

fn op_if(stack: &mut Vec<Value>) {
    // スタックから3つ取り、1つめが0なら2つめを返却、そうでなければ3つ目を返却

    let false_block = stack.pop().unwrap();
    let true_block = stack.pop().unwrap();
    let cond_block = stack.pop().unwrap();

    for value in cond_block.to_block() {
        dbg!(&value);
        eval(value, stack);
    }

    let result = stack.pop().unwrap().as_num();
    if result==0 {
        for val in true_block.to_block() {
            eval(val, stack);
        }
    } else {
        for val in false_block.to_block() {
            eval(val, stack);
        }
    }
}

fn eval<'src>(code: Value<'src>, stack: &mut Vec<Value<'src>>) {
    match code {
        Value::Op(op) => match op {
            "+" => add(stack),
            "-" => sub(stack),
            "*" => mul(stack),
            "/" => div(stack),
            _ => panic!("{op} is aaaa")
        },
        _ => stack.push(code.clone())
    };
}


fn parse<'a>(line: &'a str) -> Vec<Value<'a>>{
    let mut stack: Vec<Value> = vec!();
    let mut words: Vec<_> = line.split(" ").collect();

    while let Some((&word, mut rest)) = words.split_first() {
        dbg!(&stack, word, rest);
        if word == "{" {
            let value;
            (value, rest) = parse_block(rest);
            stack.push(value);
        } else if word == "if" {
            op_if(&mut stack);
        } else {
            let code = if let Ok(num) = word.parse::<i32>() {
                Value::Num(num)
            } else {
                Value::Op(word)
            };
            eval(code, &mut stack);
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
        } 

        let code = if let Ok(num) = word.parse::<i32>() {
            Value::Num(num)
        } else {
            Value::Op(word)
        };
        
        eval(code, &mut tokens);

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


    #[test]
    fn test_if_true() {
        assert_eq!(
            parse("{ 2 2 - } { 5 2 + } { 3 3 * } if"), 
            vec![Value::Num(7)]
        );
    }

    #[test]
    fn test_if_false() {
        assert_eq!(
            parse("{ 3 2 - } { 5 2 + } { 3 3 * } if"), 
            vec![Value::Num(9)]
        );
    }

    #[test]
    fn test_if_true_2() {
        assert_eq!(
            parse("{ 1 -1 + } { 100 } { -100 } if"), 
            vec![Value::Num(100)]
        );
    }
}