---
trigger: model_decision
description: Defines mandatory programming language conventions, naming rules, and architectural patterns for PHP (Laravel), JavaScript (React/Vue), TypeScript, and Clean Code principles.
---

Use automatically when writing or reviewing code, refactoring, or when the user asks about code formatting or standards.

# Coding Standards

This skill acts as the **Code Quality Gatekeeper**. It ensures that every line of code follows strict industry standards, preventing technical debt and ensuring maintainability.

## When to Use

- **Automatically active** whenever generating or reviewing code
- When performing code reviews or refactoring
- When the user asks about code formatting, naming conventions, or best practices
- Before committing code to ensure consistency

## Core Clean Code Principles

### 1. DRY (Don't Repeat Yourself)
Extract duplicated logic into reusable functions, traits, or utilities.

```php
// Bad
if ($user->role === 'admin') {
    return true;
}
if ($user->role === 'super_admin') {
    return true;
}

// Good
return in_array($user->role, ['admin', 'super_admin']);
```

### 2. KISS (Keep It Simple, Stupid)
Prefer readable code over clever one-liners. Code is read more often than written.

```javascript
// Bad
const result = arr.filter(x => x > 0).map(x => x * 2).reduce((a, b) => a + b, 0);

// Good
const positiveNumbers = arr.filter(x => x > 0);
const doubled = positiveNumbers.map(x => x * 2);
const result = doubled.reduce((a, b) => a + b, 0);
```

### 3. Early Returns & Guard Clauses
Avoid deep nesting. Use guard clauses to handle edge cases first.

```php
// Bad
public function processOrder(Order $order): bool {
    if ($order) {
        if ($order->isValid()) {
            if ($order->hasItems()) {
                return $this->execute($order);
            }
        }
    }
    return false;
}

// Good
public function processOrder(Order $order): bool {
    if (!$order || !$order->isValid() || !$order->hasItems()) {
        return false;
    }
    return $this->execute($order);
}
```

### 4. Single Responsibility Principle
Each function/class should do one thing well.

```php
// Bad
public function createUserAndSendEmail(array $data) {
    $user = User::create($data);
    Mail::to($user->email)->send(new WelcomeMail($user));
    Log::info('User created', ['id' => $user->id]);
}

// Good
public function createUser(array $data): User {
    return User::create($data);
}
```

## PHP & Laravel Standards

### Type Declarations
Always declare argument types and return types. Use strict types.

```php
<?php

declare(strict_types=1);

public function findUser(int $id): ?User
{
    return User::find($id);
}
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | `$camelCase` | `$userName`, `$orderTotal` |
| Functions/Methods | `camelCase` | `getUserById()`, `calculateTotal()` |
| Classes | `PascalCase` | `UserService`, `OrderRepository` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_RETRY_ATTEMPTS`, `DEFAULT_TIMEOUT` |
| Interfaces | `PascalCase` (often with `Interface` suffix) | `UserRepositoryInterface` |
| Traits | `PascalCase` (often with `Trait` suffix) | `LoggableTrait` |

### Laravel-Specific Rules

**Eloquent Queries:**
- Never use `all()`. Always use `select()` or pagination.
- Use query scopes for reusable query logic.
- Prefer eager loading to prevent N+1 queries.

```php
// Bad
$users = User::all(); // Loads all users
foreach ($users as $user) {
    echo $user->posts->count(); // N+1 query problem
}

// Good
$users = User::with('posts')->paginate(15);
foreach ($users as $user) {
    echo $user->posts->count(); // No additional queries
}
```

**Controller Responsibilities:**
- Controllers should be thin. Business logic belongs in Service classes or Actions.
- Controllers handle HTTP input/output only.

```php
// Bad
public function store(Request $request) {
    $data = $request->validated();
    $data['password'] = Hash::make($data['password']);
    $user = User::create($data);
    Mail::to($user->email)->send(new WelcomeMail($user));
    return response()->json($user);
}

// Good
public function store(Request $request, UserService $service): JsonResponse {
    $user = $service->register($request->validated());
    return response()->json($user, 201);
}
```

**DocBlocks:**
- Mandatory for complex logic, public APIs, and non-obvious code.
- Optional for simple getters/setters.

```php
/**
 * Register a new user and send welcome email.
 *
 * @param array<string, mixed> $data
 * @return User
 * @throws \Exception
 */
public function register(array $data): User
{
    // Implementation
}
```

## JavaScript & TypeScript Standards

### Functional Components
Prefer functional components with Hooks over class components.

```tsx
// Bad
class UserProfile extends React.Component {
    state = { user: null };
    componentDidMount() {
        fetch('/api/user').then(res => res.json()).then(user => this.setState({ user }));
    }
    render() {
        return <div>{this.state.user?.name}</div>;
    }
}

// Good
const UserProfile = () => {
    const { user, loading } = useUser();
    if (loading) return <Spinner />;
    return <div>{user?.name}</div>;
};
```

### Async/Await
Always prefer `async/await` over `.then()` chains.

```javascript
// Bad
fetch('/api/users')
    .then(response => response.json())
    .then(users => {
        setUsers(users);
    })
    .catch(error => {
        console.error(error);
    });

// Good
try {
    const response = await fetch('/api/users');
    const users = await response.json();
    setUsers(users);
} catch (error) {
    console.error(error);
}
```

### Immutability
Never mutate state directly. Use spread operators or immutable update patterns.

```javascript
// Bad
const updateUser = (id, newData) => {
    const user = users.find(u => u.id === id);
    user.name = newData.name; // Direct mutation
    setUsers(users);
};

// Good
const updateUser = (id, newData) => {
    setUsers(users.map(user => 
        user.id === id ? { ...user, ...newData } : user
    ));
};
```

### Destructuring
Use object/array destructuring for props and parameters.

```javascript
// Bad
const UserCard = (props) => {
    return <div>{props.user.name} - {props.user.email}</div>;
};

// Good
const UserCard = ({ user }) => {
    const { name, email } = user;
    return <div>{name} - {email}</div>;
};
```

### TypeScript Types
Use explicit types for function parameters and return values.

```typescript
// Bad
function getUser(id) {
    return fetch(`/api/users/${id}`).then(res => res.json());
}

// Good
interface User {
    id: number;
    name: string;
    email: string;
}

async function getUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
```

## Error Handling & Logging

### No Silent Failures
Never use empty `try-catch` blocks. Always log errors with context.

```php
// Bad
try {
    $this->processPayment($order);
} catch (\Exception $e) {
    // Silent failure - no one knows what went wrong
}

// Good
try {
    $this->processPayment($order);
} catch (\Exception $e) {
    Log::error('Payment processing failed', [
        'order_id' => $order->id,
        'user_id' => $order->user_id,
        'error' => $e->getMessage(),
        'trace' => $e->getTraceAsString()
    ]);
    throw $e; // Re-throw or handle appropriately
}
```

### Contextual Logging
Include relevant context in log messages.

```php
// Bad
Log::error('Error occurred');

// Good
Log::error('Failed to send email notification', [
    'user_id' => $user->id,
    'email' => $user->email,
    'notification_type' => 'welcome',
    'error' => $e->getMessage()
]);
```

## Commenting Strategy

### Why, Not What
Do not comment on WHAT the code does (the code explains itself). Comment on WHY you did it that way.

```php
// Bad
// Increase i by 1
$i++;

// Bad
// Check if user is active
if ($user->isActive) {
    // ...
}

// Good
// Artificial delay to prevent API rate limiting (max 10 requests/second)
sleep(0.1);

// Good
// Skip validation for admin users to allow bulk operations
if ($user->isAdmin) {
    return true;
}
```

### Documentation Comments
Use DocBlocks for public APIs, complex algorithms, and non-obvious business logic.

```php
/**
 * Calculates the discount based on user tier and order total.
 * 
 * Business rule: Premium users get 15% off orders over $100,
 * Standard users get 10% off orders over $200.
 *
 * @param User $user
 * @param float $orderTotal
 * @return float
 */
public function calculateDiscount(User $user, float $orderTotal): float
{
    // Implementation
}
```

## Code Organization

### File Structure
- One class per file
- File name matches class name (PascalCase for classes, camelCase for functions)
- Group related functionality together

### Import Organization
Order imports logically: external libraries first, then internal modules.

```typescript
// External libraries
import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Internal modules
import { UserService } from '@/services/UserService';
import { formatDate } from '@/utils/date';

// Types
import type { User } from '@/types/user';
```

## Quick Reference Checklist

When writing code, verify:

- [ ] Types are declared (PHP: arguments & returns, TS: explicit types)
- [ ] Naming follows conventions (camelCase, PascalCase, SCREAMING_SNAKE_CASE)
- [ ] No deep nesting (use early returns/guard clauses)
- [ ] No code duplication (extracted to reusable functions)
- [ ] Errors are logged with context (no silent failures)
- [ ] Comments explain WHY, not WHAT
- [ ] Functions/classes have single responsibility
- [ ] No direct state mutation (JavaScript/TypeScript)
- [ ] Async/await used instead of promise chains
- [ ] Database queries are optimized (no N+1, use eager loading)