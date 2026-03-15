# Identifiers (`ident`)

```
ident := <letter_or_underscore> (<letter_or_digit_or_underscore>)*
```

# Type (`type`)

```
type :=
	 <_: fn_type>
	 | <_: base_type>

base_type :=
	String
	| Int
	| Float
	| Bool

fn_type := <input : type> -> <output : type>
```

# Enum (`enum`)

```
enum :=
  enum <name: ident>
  (| <variant>)*
  end

variant :=
  <name: ident> ({ <field_name: ident> :: <type> (, <field_name: ident> :: <type>)* } )?
```

## Example

```cm
enum Animal
| Dog { age :: Int, name :: String }
| Cat { meow :: Bool }
| Fish
end
```

# Structs (`struct`)

```
struct :=
	struct <name: ident> { <field_name: ident> :: <_: type> (, <field_name: ident> :: <_: type>)* }
```

## Example

```cm
struct Person { name :: String, age :: Int, fav_function :: Int -> Int }
```

# Match

The otherwise branch needs to be included unless the compiler can be
sure that all cases have been matched, i.e it is not needed when
matching enums. Note, patterns not yet defined.

```
match :=
  match <expr> with
  (| <pat> => <expr>)*
  (| otherwise => <default : expr>)?
  end
```

## Example

```
match x with
| otherwise => x
end
```

# Choose

This is a simpler version of match, instead of matching the expression
with patterns, it can match with arbitrary boolean expressions.

Because this is a "looser" version of a match, you will always need to
have an otherwise branch, same way you always need an else when doing
an if expression.

```
choose :=
	choose
	(| <bool_expr: expr> => <expr>)*
	| otherwise => <default : expr>
	end
```

## Example

```
typ sign :: Int -> Int
def sign x =
    choose
    | x < 0 => -1
    | x == 0 => 0
    | otherwise => 1
    end
```
