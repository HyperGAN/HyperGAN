---
description: A DSL for training hyperparams that can change over time
---

# Configurable Parameters

### TODO: Not working in pytorch

## Creation

```text
gan.configurable_param(str)
```

{% hint style="info" %}
non-dsl values pass through `configurable_param` untouched
{% endhint %}

## decay

Decay from one value to another over

### Examples

```javascript
"learn_rate": "decay(range=1e-4:1e-2 steps=100000 start=300000)"
```

Increase the learn rate from 1e-4 to 1e-2, starting at step 100,000 ending at 300,000

### Options

| attribute | description | type |
| :--- | :--- | :--- |
| start | The training step to start from | Int &gt;= 0 \(default 0\) |
| steps | Number of steps until decay ends | Int &gt;= 0 \(default 10000\) |
| repeat | Repeat when complete | Boolean \(default false\) |
| metric | Reported value in stdout | String |

## anneal

$$
pow(a, t/T)
$$

### Examples

```javascript
"gamma": "anneal(0.9 T=100)"
```

### Options

| attribute | description | type |
| :--- | :--- | :--- |
| T or steps | Number of steps until decay ends | Int &gt;= 0 \(default 10000\) |
| metric | Reported value in stdout | String |

