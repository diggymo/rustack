fn main() {

    let mut stack: Vec<i32> = vec!();

    for line in std::io::stdin().lines() {
        if let Ok(_line) = line {
            let words = _line.split(" ").collect::<Vec<&str>>();
            // println!("Line: {words:?}. line: {_line}");

            for word in words {
                if let Ok(word) = word.parse::<i32>() {
                    stack.push(word)
                } else {
                    match word {
                        "+" => add(&mut stack),
                        "-" => sub(&mut stack),
                        "*" => mul(&mut stack),
                        "/" => div(&mut stack),
                        _ => panic!("{word} is aaaa")
                    }
                }
            }
        }

        println!("Stack is {stack:?}");
    }
}


fn add(stack: &mut Vec<i32>) {
    let left_hand = stack.pop().unwrap();
    let right_hand = stack.pop().unwrap();
    stack.push(left_hand+right_hand);
}

fn sub(stack: &mut Vec<i32>) {
    let right_hand = stack.pop().unwrap();
    let left_hand = stack.pop().unwrap();
    stack.push(left_hand-right_hand);
}

fn mul(stack: &mut Vec<i32>) {
    let left_hand = stack.pop().unwrap();
    let right_hand = stack.pop().unwrap();
    stack.push(right_hand*left_hand);
}

fn div(stack: &mut Vec<i32>) {
    let right_hand = stack.pop().unwrap();
    let left_hand = stack.pop().unwrap();
    stack.push(left_hand/right_hand);
}
