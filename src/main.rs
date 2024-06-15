fn main() {
    let mut stack = vec![];

    stack.push(42);
    stack.push(36);


    add(&mut stack);

    stack.push(22);

    add(&mut stack);
    
    println!("stack: {stack:?}");
}


fn add(stack: &mut Vec<i32>) {
    let left_hand = stack.pop().unwrap();
    let right_hand = stack.pop().unwrap();
    stack.push(left_hand+right_hand);
}