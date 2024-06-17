use std::{collections::HashMap, vec, io::{BufReader, BufRead}, fs::File};

#[derive(Debug)]
struct Vm {
    stack: Vec<Value>,
    vars: HashMap<String, Value>,
    blocks: Vec<Vec<Value>>
}

impl Vm {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            vars: HashMap::new(),
            blocks: vec![]
        }
    }

    pub fn get_current_scope_stack(&mut self) -> &mut Vec<Value>{
        return if self.blocks.len() == 0 {
            &mut self.stack
        } else {
            self.blocks.last_mut().unwrap()
        };
    
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Value {
    Num(i32),
    Op(String),
    Sym(String),
    Block(Vec<Value>)
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
fn parse_batch(reader: impl BufRead) -> Vec<Value> {
    let mut vm = Vm::new();

    for line in reader.lines().flatten() {
        for word in line.split(" ") {
            parse_word(word, &mut vm);
        }
        // parse(line, &mut vm);
    }

    vm.stack
    // if let Ok(_) = reader.read_line(&mut buf) {
    //     parse(buf.clone(), &mut vm);
    // }
}

fn parse_interactive() {
    let mut vm = Vm::new();
    for line in std::io::stdin().lines().flatten() {
        // parse(line.clone(), &mut vm);
        for word in line.split(" ") {
            parse_word(word, &mut vm);
        }
        println!("stack: {:?}", vm.stack);
    }
}

fn parse_word(word: &str, vm: &mut Vm) {
    if word.is_empty() {
        return;
    }

    if word == "{" {
        vm.blocks.push(vec![]);
        // let value;
        // (value, rest) = parse_block(rest, vm);
        // vm.stack.push(value);
    } else if word == "}" {
        if let Some(last_block) = vm.blocks.pop() {
            eval(Value::Block(last_block), vm);
            // vm.stack.push(Value::Block(last_block));
        } else {
            panic!("block is empty...");
        }
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
}

fn op_if(vm: &mut Vm) {
    // スタックから3つ取り、1つめが0なら2つめを返却、そうでなければ3つ目を返却
    let stack = vm.get_current_scope_stack();

    let false_block = stack.pop().unwrap();
    let true_block = stack.pop().unwrap();
    let cond_block = stack.pop().unwrap();

    for value in cond_block.to_block() {
        eval(value, vm);
    }

    let stack = vm.get_current_scope_stack();

    let result = stack.pop().unwrap().as_num();
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
            "+" => add(vm),
            "-" => sub(vm),
            "*" => mul(vm),
            "/" => div(vm),
            "<" => lt(vm),
            "if" => op_if(vm),
            "def" => opt_def(vm),
            "puts" => puts(&vm.stack),
            _ => {
                let val = vm
                    .vars
                    .get(op.as_str())
                    .expect(&format!("{op:?} is not defined")).clone();
                
                let stack = vm.get_current_scope_stack();
                stack.push(val);
            }
        },
        _ => {
            let stack = vm.get_current_scope_stack();
            stack.push(code.clone())
        },
    };
}

fn opt_def(vm: &mut Vm) {
    let stack = vm.get_current_scope_stack();

    let right_hand = stack.pop().unwrap();
    let left_hand = stack.pop().unwrap();

    if let Value::Sym(sym) = left_hand {
        vm.vars.insert(sym, right_hand);
    }
}

macro_rules! impl_op {
    ($name:ident, $op:tt) => {
        fn $name(vm: &mut Vm) {
            let stack = vm.get_current_scope_stack();
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
        let result = parse_batch(Cursor::new(" 1 2 + 2"));
        assert_eq!(result, vec![Value::Num(3), Value::Num(2)]);
    }

    #[test]
    fn test_sub() {
        let result = parse_batch(Cursor::new("2 2 -"));
        assert_eq!(result, vec![Value::Num(0)]);
    }

    #[test]
    fn test_group() {
        let result = parse_batch(Cursor::new("1 2 + { 3 4 }"));
        assert_eq!(
            result,
            vec![
                Value::Num(3),
                Value::Block(vec![Value::Num(3), Value::Num(4)])
            ]
        );
    }

    #[test]
    fn test_if_true() {
        let result = parse_batch(Cursor::new("{ 2 2 - } { 5 2 + } { 3 3 * } if"));
        assert_eq!(result, vec![Value::Num(9)]);
    }

    #[test]
    fn test_if_false() {
        let result = parse_batch(Cursor::new("{ 3 2 - } { 5 2 + } { 3 3 * } if"));
        assert_eq!(result, vec![Value::Num(7)]);
    }

    #[test]
    fn test_variables_add() {
        let result = parse_batch(Cursor::new("/x 10 def /y 20 def y x +"));
        assert_eq!(result, vec![Value::Num(30)]);
    }

    #[test]
    fn test_variables_if() {
        let result = parse_batch(Cursor::new("/x 10 def /y 20 def { x y < } { x } { y } if"));
        assert_eq!(result, vec![Value::Num(10)]);
    }


    #[test]
    fn test_puts() {
        let result = parse_batch(Cursor::new("/x 10 def /y 20 def x y + puts"));
        assert_eq!(result, vec![Value::Num(30)]);
    }

    #[test]
    fn test_parse_with_sometimes() {
        let result = parse_batch(Cursor::new("/x 10 def /y 20 def x y +\n15 2 * -"));
        assert_eq!(result, vec![Value::Num(0)]);
    }
}
