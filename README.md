## 「Rustでつくるプログラミング言語」の実装ログ

```
cargo test

cargo run --bin main_2

cargo run --bin main_2 -- ./example_scripts/factorial.txt

cargo run --bin main_2 -- ./example_scripts/fibonacci.txt

```

```
cargo run --bin rustack < ./example_scripts/ruscal.txt

echo "1+2;" | cargo run --bin rustack
```

![](./architecture.drawio.svg)
