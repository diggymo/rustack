use std::{collections::HashMap, vec, io::{BufReader, BufRead}, fs::File};

#[derive(Debug)]
struct Vm {
    stack: Vec<Value>,
    vars: HashMap<String, Value>,
}

impl Vm {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            vars: HashMap::new(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Value {
    Num(i32),
    Op(String),
    Block(Vec<Value>),
    Sym(String),
}

impl Value {
    fn as_num(&self) -> i32 {
        match self {
            Self::Num(i) => *i,
            _ => panic!("{self:?} is not a number..."),
        }
    }

    fn to_block(self) -> Vec<Value> {
        match self {
            Self::Block(val) => val,
            _ => panic!("###"),
        }
    }

    fn to_string(&self) -> String {
        match self {
            Self::Num(i) => i.to_string(),
            Self::Op(v) | Self::Sym(v) => v.clone(),
            Self::Block(_) => "<Block>".into()
        }
    }
}

fn puts(stack: &Vec<Value>) {
    for v in stack {
        println!("{}", v.to_string());
    }
}

fn main() {
    if let Some(f) = std::env::args().nth(1).and_then(|f| std::fs::File::open(f).ok()) {
        parse_batch(BufReader::new(f));
    } else {
        parse_interactive();
    }

}

/** `BufReader<File>` は `impl BufRead`になる */
fn parse_batch(reader: impl BufRead) {
    let mut vm = Vm::new();

    for line in reader.lines().flatten() {
        parse(line, &mut vm);
    }
    // if let Ok(_) = reader.read_line(&mut buf) {
    //     parse(buf.clone(), &mut vm);
    // }
}

fn parse_interactive() {
    let mut vm = Vm::new();
    for line in std::io::stdin().lines().flatten() {
        parse(line.clone(), &mut vm);
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

    let result = vm.stack.pop().unwrap().as_num();
    if result != 0 {
        for val in true_block.to_block() {
            eval(val, vm);
        }
    } else {
        for val in false_block.to_block() {
            eval(val, vm);
        }
    }
}

fn eval(code: Value, vm: &mut Vm) {
    match code {
        Value::Op(op) => match op.as_str() {
            "+" => add(&mut vm.stack),
            "-" => sub(&mut vm.stack),
            "*" => mul(&mut vm.stack),
            "/" => div(&mut vm.stack),
            "<" => lt(&mut vm.stack),
            "if" => op_if(vm),
            "def" => opt_def(vm),
            "puts" => puts(&vm.stack),
            _ => {
                let val = vm
                    .vars
                    .get(op.as_str())
                    .expect(&format!("{op:?} is not defined"));
                vm.stack.push(val.clone());
            }
        },
        _ => vm.stack.push(code.clone()),
    };
}

/** FIXME: evalと共通化したい...が、独自のstackを持つ必要があるのでできない... */
fn eval_in_block(code: Value, stack: &mut Vec<Value>, vars: &mut HashMap<String, Value>) {
    match code {
        Value::Op(op) => match op.as_str() {
            "+" => add(stack),
            "-" => sub(stack),
            "*" => mul(stack),
            "/" => div(stack),
            "<" => lt(stack),
            "puts" => puts(stack),
            _ => {
                let val = vars
                    .get(op.as_str())
                    .expect(&format!("{op:?} is not defined"));
                stack.push(val.clone());
            }
        },
        _ => stack.push(code.clone()),
    };
}

fn parse(line: String, vm: &mut Vm) {
    let input: Vec<_> = line.split(" ").collect();
    // NOTE: Vec<&str> → &[&str]への変換
    let mut words = &input[..];

    while let Some((&word, mut rest)) = words.split_first() {
        if word.is_empty() {
            continue;
        }

        if word == "{" {
            let value;
            (value, rest) = parse_block(rest, vm);
            vm.stack.push(value);
        } else {
            let code = if let Ok(num) = word.parse::<i32>() {
                Value::Num(num)
            } else if word.starts_with("/") {
                Value::Sym(word[1..].to_string())
            } else {
                Value::Op(word.into())
            };
            eval(code, vm);
        }

        words = rest;
    }
}

fn parse_block<'src, 'a>(input: &'a [&'src str], vm: &mut Vm) -> (Value, &'a [&'src str]) {
    let mut tokens = vec![];
    let mut words = input;

    while let Some((&word, mut rest)) = words.split_first() {
        if word.is_empty() {
            break;
        }

        if word == "{" {
            let value;
            (value, rest) = parse_block(rest, vm);

            tokens.push(value);
        } else if word == "}" {
            return (Value::Block(tokens), rest);
        }

        let code = if let Ok(num) = word.parse::<i32>() {
            Value::Num(num)
        } else {
            Value::Op(word.into())
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
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_blck() {
        let result = parse_block(&["1", "2", "+", "}", "2"], &mut Vm::new());
        assert_eq!(result.0, Value::Block(vec![Value::Num(3)]));
        assert_eq!(result.1, &["2"]);
    }

    #[test]
    fn test_sub() {
        let mut vm = Vm::new();
        parse("2 2 -".into(), &mut vm);
        assert_eq!(vm.stack, vec![Value::Num(0)]);
    }

    #[test]
    fn test_group() {
        let mut vm = Vm::new();
        parse("1 2 + { 3 4 }".into(), &mut vm);
        assert_eq!(
            vm.stack,
            vec![
                Value::Num(3),
                Value::Block(vec![Value::Num(3), Value::Num(4)])
            ]
        );
    }

    #[test]
    fn test_if_true() {
        let mut vm = Vm::new();
        parse("{ 2 2 - } { 5 2 + } { 3 3 * } if".into(), &mut vm);
        assert_eq!(vm.stack, vec![Value::Num(9)]);
    }

    #[test]
    fn test_if_false() {
        let mut vm = Vm::new();
        parse("{ 3 2 - } { 5 2 + } { 3 3 * } if".into(), &mut vm);
        assert_eq!(vm.stack, vec![Value::Num(7)]);
    }

    #[test]
    fn test_variables_add() {
        let mut vm = Vm::new();
        parse("/x 10 def /y 20 def y x +".into(), &mut vm);
        assert_eq!(vm.stack, vec![Value::Num(30)]);
    }

    #[test]
    fn test_variables_if() {
        let mut vm = Vm::new();
        parse(
            "/x 10 def /y 20 def { x y < } { x } { y } if".into(),
            &mut vm,
        );
        assert_eq!(vm.stack, vec![Value::Num(10)]);
    }


    #[test]
    fn test_puts() {
        let mut vm = Vm::new();
        parse(
            "/x 10 def /y 20 def x y + puts".into(),
            &mut vm,
        );
        assert_eq!(vm.stack, vec![Value::Num(30)]);
    }

    #[test]
    fn test_parse_with_sometimes() {
        let mut vm = Vm::new();
        parse(
            "/x 10 def /y 20 def x y +".into(),
            &mut vm,
        );
        dbg!(&vm.stack);
        parse(
            "15 2 * -".into(),
            &mut vm,
        );

        assert_eq!(vm.stack, vec![Value::Num(0)]);
    }
}
