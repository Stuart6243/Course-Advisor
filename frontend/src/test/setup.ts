import '../i18n';

Object.defineProperty(Element.prototype, 'scrollIntoView', {
  configurable: true,
  value: () => undefined,
});

Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
