use std::{collections::HashMap, vec};

#[derive(Debug)]
struct Vm<'src> {
    stack: Vec<Value<'src>>,
    vars: HashMap<&'src str, Value<'src>>
}

impl<'src> Vm<'src> {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            vars: HashMap::new()
        }
    }
}


#[derive(Debug, PartialEq, Eq, Clone)]
enum Value<'src> {
    Num(i32),
    Op(&'src str),
    Block(Vec<Value<'src>>),
    Sym(&'src str)
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

fn op_if(vm: &mut Vm) {
    // スタックから3つ取り、1つめが0なら2つめを返却、そうでなければ3つ目を返却

    let false_block = vm.stack.pop().unwrap();
    let true_block = vm.stack.pop().unwrap();
    let cond_block = vm.stack.pop().unwrap();

    for value in cond_block.to_block() {
        eval(value, vm);
    }

    dbg!(&vm.stack);
    let result = vm.stack.pop().unwrap().as_num();
    if result!=0 {
        for val in true_block.to_block() {
            eval(val, vm);
        }
    } else {
        for val in false_block.to_block() {
            eval(val, vm);
        }
    }
}

fn eval<'src>(code: Value<'src>, vm: &mut Vm<'src>) {
    match code {
        Value::Op(op) => match op {
            "+" => add(&mut vm.stack),
            "-" => sub(&mut vm.stack),
            "*" => mul(&mut vm.stack),
            "/" => div(&mut vm.stack),
            "<" => {
                dbg!(&vm.stack);
                lt(&mut vm.stack);
                dbg!(&vm.stack);
            },
            "if" => op_if(vm),
            "def" => opt_def(vm),
            _ => {
                dbg!(&vm.vars);
                let val = vm.vars.get(op).expect(&format!("{op:?} is not defined"));
                vm.stack.push(val.clone());
            }
        },
        _ => vm.stack.push(code.clone())
    };
}

/** FIXME: evalと共通化したい...が、独自のstackを持つ必要があるのでできない... */
fn eval_in_block<'src>(code: Value<'src>, stack: &mut Vec<Value<'src>>, vars: &mut HashMap<&str, Value<'src>>) {
    match code {
        Value::Op(op) => match op {
            "+" => add(stack),
            "-" => sub(stack),
            "*" => mul(stack),
            "/" => div(stack),
            "<" => lt(stack),
            _ => {
                let val = vars.get(op).expect(&format!("{op:?} is not defined"));
                stack.push(val.clone());
            }
        },
        _ => stack.push(code.clone())
    };
}


fn parse<'a>(line: &'a str) -> Vec<Value<'a>>{
    let mut vm = Vm::new();

    let input: Vec<_> = line.split(" ").collect();
    // NOTE: Vec<&str> → &[&str]への変換
    let mut words = &input[..];

    while let Some((&word, mut rest)) = words.split_first() {
        if word.is_empty() {
            continue;
        }
        
        if word == "{" {
            let value;
            (value, rest) = parse_block(rest, &mut vm);
            vm.stack.push(value);
        } else {
            let code = if let Ok(num) = word.parse::<i32>() {
                Value::Num(num)
            } else if word.starts_with("/") {
                Value::Sym(&word[1..])
            } else {
                Value::Op(word)
            };
            eval(code, &mut vm);
        }

        words = rest;
    }

    return vm.stack;
}

fn parse_block<'src, 'a>(input: &'a [&'src str],vm: &mut Vm<'src>) -> (Value<'src>, &'a [&'src str]) {
    let mut tokens = vec![];
    let mut words = input;
    
    while let Some((&word, mut rest)) = words.split_first() {
        if word.is_empty() {
            break;
        }

        if word == "{" {
            let value;
            (value, rest) = parse_block(rest, vm);
            dbg!(&value, &rest, &tokens);
            
            tokens.push(value);
        } else if word == "}" {
            return (Value::Block(tokens), rest);
        } 

        let code = if let Ok(num) = word.parse::<i32>() {
            Value::Num(num)
        } else {
            Value::Op(word)
        };
        
        eval_in_block(code, &mut tokens, &mut vm.vars);

        words = rest;
    }

    (Value::Block(tokens), words)
}

fn opt_def(vm: &mut Vm) {
    let right_hand = vm.stack.pop().unwrap();
    let left_hand = vm.stack.pop().unwrap();

    if let Value::Sym(sym) = left_hand {
        vm.vars.insert(sym, right_hand);
    }
}


macro_rules! impl_op {
    ($name:ident, $op:tt) => {
        fn $name(stack: &mut Vec<Value>) {
            let right_hand = stack.pop().unwrap().as_num();
            let left_hand = stack.pop().unwrap().as_num();
            stack.push(Value::Num((left_hand $op right_hand) as i32));
        }
    };
}

impl_op!(add, +);
impl_op!(sub, -);
impl_op!(mul, *);
impl_op!(div, /);
impl_op!(lt, <);

#[cfg(test)]
mod test {
    use super::*;


    #[test]
    fn test_blck() {
        let result = parse_block(&["1", "2", "+", "}", "2"], &mut Vm::new());
        assert_eq!(
            result.0,
            Value::Block(vec![Value::Num(3)])
        );
        assert_eq!(
            result.1,
            &["2"]
        );
    }


    #[test]
    fn test_sub() {
        assert_eq!(
            parse("2 2 -"), 
            vec![Value::Num(0)]
        );
    }


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
            vec![Value::Num(9)]
        );
    }

    #[test]
    fn test_if_false() {
        assert_eq!(
            parse("{ 3 2 - } { 5 2 + } { 3 3 * } if"), 
            vec![Value::Num(7)]
        );
    }

    #[test]
    fn test_variables_add() {
        assert_eq!(
            parse("/x 10 def /y 20 def y x +"), 
            vec![Value::Num(30)]
        );
    }

    #[test]
    fn test_variables_if() {
        assert_eq!(
            parse("/x 10 def /y 20 def { x y < } { x } { y } if"), 
            vec![Value::Num(10)]
        );
    }

}