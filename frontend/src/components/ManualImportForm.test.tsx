import {describe, expect, it} from 'vitest';

import {
  isValidManualCourseCode,
  normalizeManualTerm,
  parseManualPoints,
} from './ManualImportForm';

describe('parseManualPoints', () => {
  it.each([
    ['3', {min: 3, max: 3}],
    ['3.5', {min: 3.5, max: 3.5}],
    ['1 - 4', {min: 1, max: 4}],
  ])('accepts valid points %s', (raw, expected) => {
    expect(parseManualPoints(raw)).toEqual(expected);
  });

  it.each(['', 'zero', '0', '-1', '4-2', '3-', '2-31'])('rejects invalid points %s', (raw) => {
    expect(parseManualPoints(raw)).toBeNull();
  });
});

describe('normalizeManualTerm', () => {
  it('normalizes a valid syllabus term', () => {
    expect(normalizeManualTerm('  spring   2026 ')).toBe('Spring 2026');
  });

  it.each(['', '2026 Spring', 'Autumn 2026', 'Spring 26'])('rejects %s', (raw) => {
    expect(normalizeManualTerm(raw)).toBeNull();
  });
});

describe('isValidManualCourseCode', () => {
  it.each(['COMS W4111', 'PSAM UN3707', 'BINF GU4001', 'EESC GR5400'])(
    'accepts current catalog level %s',
    (code) => {
      expect(isValidManualCourseCode(code)).toBe(true);
    },
  );

  it.each(['COMS ZZ4111', 'COMS U4111X', 'COMS 4111'])(
    'rejects unsupported level %s',
    (code) => {
      expect(isValidManualCourseCode(code)).toBe(false);
    },
  );
});
