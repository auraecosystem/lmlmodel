import std.stdio;
import core.stdc.stdlib : malloc, free, alloca;
import std.exception : assertThrown, assertNotThrown;

// ==========================================
// Compile-Time Validation (static assert)
// ==========================================

// Guarantees binary layouts and platform assumptions at compile time
static assert(float.sizeof == 4, "Float size must be exactly 4 bytes for raw slicing.");
static assert(uint.max == 0xFFFF_FFFF, "uint must be 32 bits wide.");

void livingDangerously()
{
    enum bytes = float.sizeof * 1024 * 1024;
    auto buf = malloc(bytes);
    if (!buf)
        throw new Exception("Memory allocation failed");

    scope(exit) free(buf);

    auto floats = cast(float[]) buf[0 .. bytes];
    floats[0] = 3.14159f;

    enum stackBytes = 4096 * 10;
    void* stackBuf = alloca(stackBytes);
    if (stackBuf)
    {
        auto stackInts = cast(int[]) stackBuf[0 .. stackBytes];
        stackInts[0] = 42;
    }
}

// Unit test placed directly next to the function it tests
unittest
{
    // Verifies memory allocation routine completes without throwing exceptions
    assertNotThrown(livingDangerously());
}

uint checked_multiply(uint x, uint y)
{
    uint result;

    version (D_InlineAsm_X86)
    {
        asm
        {
            mov     EAX, x      ;
            mul     y           ;
            mov     result, EAX ;
            jc      Loverflow   ;
        }
        return result;
    }
    else version (D_InlineAsm_X86_64)
    {
        asm
        {
            mov     EAX, x      ;
            mul     y           ;
            mov     result, EAX ;
            jc      Loverflow   ;
        }
        return result;
    }
    else
    {
        result = x * y;
        if (!y || x <= uint.max / y)
            return result;
    }

Loverflow:
    throw new Exception("Multiply overflow occurred");
}

// Comprehensive unit tests for checked_multiply
unittest
{
    // 1. Basic arithmetic validation
    assert(checked_multiply(0, 100) == 0);
    assert(checked_multiply(1200, 3500) == 4_200_000);
    assert(checked_multiply(1, uint.max) == uint.max);

    // 2. Exception testing with std.exception.assertThrown
    assertThrown!Exception(checked_multiply(uint.max, 2));
    assertThrown!Exception(checked_multiply(0x8000_0000, 2));

    // 3. Edge-case non-overflow validation
    assertNotThrown(checked_multiply(65535, 65536)); // Largest non-overflow product (0xFFFF0000)
}

void main()
{
    writeln("Application executed successfully.");
}
