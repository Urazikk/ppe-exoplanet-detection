(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))i(r);new MutationObserver(r=>{for(const s of r)if(s.type==="childList")for(const o of s.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&i(o)}).observe(document,{childList:!0,subtree:!0});function n(r){const s={};return r.integrity&&(s.integrity=r.integrity),r.referrerPolicy&&(s.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?s.credentials="include":r.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function i(r){if(r.ep)return;r.ep=!0;const s=n(r);fetch(r.href,s)}})();function Hx(t){return t&&t.__esModule&&Object.prototype.hasOwnProperty.call(t,"default")?t.default:t}var T0={exports:{}},pc={},w0={exports:{}},et={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var oa=Symbol.for("react.element"),Gx=Symbol.for("react.portal"),Wx=Symbol.for("react.fragment"),Xx=Symbol.for("react.strict_mode"),qx=Symbol.for("react.profiler"),$x=Symbol.for("react.provider"),Kx=Symbol.for("react.context"),Yx=Symbol.for("react.forward_ref"),Zx=Symbol.for("react.suspense"),Jx=Symbol.for("react.memo"),Qx=Symbol.for("react.lazy"),uh=Symbol.iterator;function ev(t){return t===null||typeof t!="object"?null:(t=uh&&t[uh]||t["@@iterator"],typeof t=="function"?t:null)}var C0={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},A0=Object.assign,R0={};function Zs(t,e,n){this.props=t,this.context=e,this.refs=R0,this.updater=n||C0}Zs.prototype.isReactComponent={};Zs.prototype.setState=function(t,e){if(typeof t!="object"&&typeof t!="function"&&t!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,t,e,"setState")};Zs.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this,t,"forceUpdate")};function I0(){}I0.prototype=Zs.prototype;function jf(t,e,n){this.props=t,this.context=e,this.refs=R0,this.updater=n||C0}var Vf=jf.prototype=new I0;Vf.constructor=jf;A0(Vf,Zs.prototype);Vf.isPureReactComponent=!0;var dh=Array.isArray,P0=Object.prototype.hasOwnProperty,Hf={current:null},D0={key:!0,ref:!0,__self:!0,__source:!0};function L0(t,e,n){var i,r={},s=null,o=null;if(e!=null)for(i in e.ref!==void 0&&(o=e.ref),e.key!==void 0&&(s=""+e.key),e)P0.call(e,i)&&!D0.hasOwnProperty(i)&&(r[i]=e[i]);var a=arguments.length-2;if(a===1)r.children=n;else if(1<a){for(var c=Array(a),u=0;u<a;u++)c[u]=arguments[u+2];r.children=c}if(t&&t.defaultProps)for(i in a=t.defaultProps,a)r[i]===void 0&&(r[i]=a[i]);return{$$typeof:oa,type:t,key:s,ref:o,props:r,_owner:Hf.current}}function tv(t,e){return{$$typeof:oa,type:t.type,key:e,ref:t.ref,props:t.props,_owner:t._owner}}function Gf(t){return typeof t=="object"&&t!==null&&t.$$typeof===oa}function nv(t){var e={"=":"=0",":":"=2"};return"$"+t.replace(/[=:]/g,function(n){return e[n]})}var fh=/\/+/g;function zc(t,e){return typeof t=="object"&&t!==null&&t.key!=null?nv(""+t.key):e.toString(36)}function hl(t,e,n,i,r){var s=typeof t;(s==="undefined"||s==="boolean")&&(t=null);var o=!1;if(t===null)o=!0;else switch(s){case"string":case"number":o=!0;break;case"object":switch(t.$$typeof){case oa:case Gx:o=!0}}if(o)return o=t,r=r(o),t=i===""?"."+zc(o,0):i,dh(r)?(n="",t!=null&&(n=t.replace(fh,"$&/")+"/"),hl(r,e,n,"",function(u){return u})):r!=null&&(Gf(r)&&(r=tv(r,n+(!r.key||o&&o.key===r.key?"":(""+r.key).replace(fh,"$&/")+"/")+t)),e.push(r)),1;if(o=0,i=i===""?".":i+":",dh(t))for(var a=0;a<t.length;a++){s=t[a];var c=i+zc(s,a);o+=hl(s,e,n,c,r)}else if(c=ev(t),typeof c=="function")for(t=c.call(t),a=0;!(s=t.next()).done;)s=s.value,c=i+zc(s,a++),o+=hl(s,e,n,c,r);else if(s==="object")throw e=String(t),Error("Objects are not valid as a React child (found: "+(e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e)+"). If you meant to render a collection of children, use an array instead.");return o}function ya(t,e,n){if(t==null)return t;var i=[],r=0;return hl(t,i,"","",function(s){return e.call(n,s,r++)}),i}function iv(t){if(t._status===-1){var e=t._result;e=e(),e.then(function(n){(t._status===0||t._status===-1)&&(t._status=1,t._result=n)},function(n){(t._status===0||t._status===-1)&&(t._status=2,t._result=n)}),t._status===-1&&(t._status=0,t._result=e)}if(t._status===1)return t._result.default;throw t._result}var un={current:null},ml={transition:null},rv={ReactCurrentDispatcher:un,ReactCurrentBatchConfig:ml,ReactCurrentOwner:Hf};function F0(){throw Error("act(...) is not supported in production builds of React.")}et.Children={map:ya,forEach:function(t,e,n){ya(t,function(){e.apply(this,arguments)},n)},count:function(t){var e=0;return ya(t,function(){e++}),e},toArray:function(t){return ya(t,function(e){return e})||[]},only:function(t){if(!Gf(t))throw Error("React.Children.only expected to receive a single React element child.");return t}};et.Component=Zs;et.Fragment=Wx;et.Profiler=qx;et.PureComponent=jf;et.StrictMode=Xx;et.Suspense=Zx;et.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=rv;et.act=F0;et.cloneElement=function(t,e,n){if(t==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+t+".");var i=A0({},t.props),r=t.key,s=t.ref,o=t._owner;if(e!=null){if(e.ref!==void 0&&(s=e.ref,o=Hf.current),e.key!==void 0&&(r=""+e.key),t.type&&t.type.defaultProps)var a=t.type.defaultProps;for(c in e)P0.call(e,c)&&!D0.hasOwnProperty(c)&&(i[c]=e[c]===void 0&&a!==void 0?a[c]:e[c])}var c=arguments.length-2;if(c===1)i.children=n;else if(1<c){a=Array(c);for(var u=0;u<c;u++)a[u]=arguments[u+2];i.children=a}return{$$typeof:oa,type:t.type,key:r,ref:s,props:i,_owner:o}};et.createContext=function(t){return t={$$typeof:Kx,_currentValue:t,_currentValue2:t,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},t.Provider={$$typeof:$x,_context:t},t.Consumer=t};et.createElement=L0;et.createFactory=function(t){var e=L0.bind(null,t);return e.type=t,e};et.createRef=function(){return{current:null}};et.forwardRef=function(t){return{$$typeof:Yx,render:t}};et.isValidElement=Gf;et.lazy=function(t){return{$$typeof:Qx,_payload:{_status:-1,_result:t},_init:iv}};et.memo=function(t,e){return{$$typeof:Jx,type:t,compare:e===void 0?null:e}};et.startTransition=function(t){var e=ml.transition;ml.transition={};try{t()}finally{ml.transition=e}};et.unstable_act=F0;et.useCallback=function(t,e){return un.current.useCallback(t,e)};et.useContext=function(t){return un.current.useContext(t)};et.useDebugValue=function(){};et.useDeferredValue=function(t){return un.current.useDeferredValue(t)};et.useEffect=function(t,e){return un.current.useEffect(t,e)};et.useId=function(){return un.current.useId()};et.useImperativeHandle=function(t,e,n){return un.current.useImperativeHandle(t,e,n)};et.useInsertionEffect=function(t,e){return un.current.useInsertionEffect(t,e)};et.useLayoutEffect=function(t,e){return un.current.useLayoutEffect(t,e)};et.useMemo=function(t,e){return un.current.useMemo(t,e)};et.useReducer=function(t,e,n){return un.current.useReducer(t,e,n)};et.useRef=function(t){return un.current.useRef(t)};et.useState=function(t){return un.current.useState(t)};et.useSyncExternalStore=function(t,e,n){return un.current.useSyncExternalStore(t,e,n)};et.useTransition=function(){return un.current.useTransition()};et.version="18.3.1";w0.exports=et;var Z=w0.exports;const sv=Hx(Z);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var ov=Z,av=Symbol.for("react.element"),lv=Symbol.for("react.fragment"),cv=Object.prototype.hasOwnProperty,uv=ov.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,dv={key:!0,ref:!0,__self:!0,__source:!0};function N0(t,e,n){var i,r={},s=null,o=null;n!==void 0&&(s=""+n),e.key!==void 0&&(s=""+e.key),e.ref!==void 0&&(o=e.ref);for(i in e)cv.call(e,i)&&!dv.hasOwnProperty(i)&&(r[i]=e[i]);if(t&&t.defaultProps)for(i in e=t.defaultProps,e)r[i]===void 0&&(r[i]=e[i]);return{$$typeof:av,type:t,key:s,ref:o,props:r,_owner:uv.current}}pc.Fragment=lv;pc.jsx=N0;pc.jsxs=N0;T0.exports=pc;var l=T0.exports,qu={},U0={exports:{}},Cn={},k0={exports:{}},O0={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(t){function e(j,$){var Q=j.length;j.push($);e:for(;0<Q;){var se=Q-1>>>1,ae=j[se];if(0<r(ae,$))j[se]=$,j[Q]=ae,Q=se;else break e}}function n(j){return j.length===0?null:j[0]}function i(j){if(j.length===0)return null;var $=j[0],Q=j.pop();if(Q!==$){j[0]=Q;e:for(var se=0,ae=j.length,Ae=ae>>>1;se<Ae;){var De=2*(se+1)-1,Oe=j[De],D=De+1,q=j[D];if(0>r(Oe,Q))D<ae&&0>r(q,Oe)?(j[se]=q,j[D]=Q,se=D):(j[se]=Oe,j[De]=Q,se=De);else if(D<ae&&0>r(q,Q))j[se]=q,j[D]=Q,se=D;else break e}}return $}function r(j,$){var Q=j.sortIndex-$.sortIndex;return Q!==0?Q:j.id-$.id}if(typeof performance=="object"&&typeof performance.now=="function"){var s=performance;t.unstable_now=function(){return s.now()}}else{var o=Date,a=o.now();t.unstable_now=function(){return o.now()-a}}var c=[],u=[],p=1,h=null,f=3,g=!1,x=!1,M=!1,v=typeof setTimeout=="function"?setTimeout:null,d=typeof clearTimeout=="function"?clearTimeout:null,m=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function _(j){for(var $=n(u);$!==null;){if($.callback===null)i(u);else if($.startTime<=j)i(u),$.sortIndex=$.expirationTime,e(c,$);else break;$=n(u)}}function b(j){if(M=!1,_(j),!x)if(n(c)!==null)x=!0,G(w);else{var $=n(u);$!==null&&z(b,$.startTime-j)}}function w(j,$){x=!1,M&&(M=!1,d(y),y=-1),g=!0;var Q=f;try{for(_($),h=n(c);h!==null&&(!(h.expirationTime>$)||j&&!I());){var se=h.callback;if(typeof se=="function"){h.callback=null,f=h.priorityLevel;var ae=se(h.expirationTime<=$);$=t.unstable_now(),typeof ae=="function"?h.callback=ae:h===n(c)&&i(c),_($)}else i(c);h=n(c)}if(h!==null)var Ae=!0;else{var De=n(u);De!==null&&z(b,De.startTime-$),Ae=!1}return Ae}finally{h=null,f=Q,g=!1}}var A=!1,E=null,y=-1,C=5,P=-1;function I(){return!(t.unstable_now()-P<C)}function F(){if(E!==null){var j=t.unstable_now();P=j;var $=!0;try{$=E(!0,j)}finally{$?B():(A=!1,E=null)}}else A=!1}var B;if(typeof m=="function")B=function(){m(F)};else if(typeof MessageChannel<"u"){var W=new MessageChannel,V=W.port2;W.port1.onmessage=F,B=function(){V.postMessage(null)}}else B=function(){v(F,0)};function G(j){E=j,A||(A=!0,B())}function z(j,$){y=v(function(){j(t.unstable_now())},$)}t.unstable_IdlePriority=5,t.unstable_ImmediatePriority=1,t.unstable_LowPriority=4,t.unstable_NormalPriority=3,t.unstable_Profiling=null,t.unstable_UserBlockingPriority=2,t.unstable_cancelCallback=function(j){j.callback=null},t.unstable_continueExecution=function(){x||g||(x=!0,G(w))},t.unstable_forceFrameRate=function(j){0>j||125<j?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):C=0<j?Math.floor(1e3/j):5},t.unstable_getCurrentPriorityLevel=function(){return f},t.unstable_getFirstCallbackNode=function(){return n(c)},t.unstable_next=function(j){switch(f){case 1:case 2:case 3:var $=3;break;default:$=f}var Q=f;f=$;try{return j()}finally{f=Q}},t.unstable_pauseExecution=function(){},t.unstable_requestPaint=function(){},t.unstable_runWithPriority=function(j,$){switch(j){case 1:case 2:case 3:case 4:case 5:break;default:j=3}var Q=f;f=j;try{return $()}finally{f=Q}},t.unstable_scheduleCallback=function(j,$,Q){var se=t.unstable_now();switch(typeof Q=="object"&&Q!==null?(Q=Q.delay,Q=typeof Q=="number"&&0<Q?se+Q:se):Q=se,j){case 1:var ae=-1;break;case 2:ae=250;break;case 5:ae=1073741823;break;case 4:ae=1e4;break;default:ae=5e3}return ae=Q+ae,j={id:p++,callback:$,priorityLevel:j,startTime:Q,expirationTime:ae,sortIndex:-1},Q>se?(j.sortIndex=Q,e(u,j),n(c)===null&&j===n(u)&&(M?(d(y),y=-1):M=!0,z(b,Q-se))):(j.sortIndex=ae,e(c,j),x||g||(x=!0,G(w))),j},t.unstable_shouldYield=I,t.unstable_wrapCallback=function(j){var $=f;return function(){var Q=f;f=$;try{return j.apply(this,arguments)}finally{f=Q}}}})(O0);k0.exports=O0;var fv=k0.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var pv=Z,wn=fv;function fe(t){for(var e="https://reactjs.org/docs/error-decoder.html?invariant="+t,n=1;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var z0=new Set,ko={};function qr(t,e){Os(t,e),Os(t+"Capture",e)}function Os(t,e){for(ko[t]=e,t=0;t<e.length;t++)z0.add(e[t])}var Pi=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),$u=Object.prototype.hasOwnProperty,hv=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,ph={},hh={};function mv(t){return $u.call(hh,t)?!0:$u.call(ph,t)?!1:hv.test(t)?hh[t]=!0:(ph[t]=!0,!1)}function gv(t,e,n,i){if(n!==null&&n.type===0)return!1;switch(typeof e){case"function":case"symbol":return!0;case"boolean":return i?!1:n!==null?!n.acceptsBooleans:(t=t.toLowerCase().slice(0,5),t!=="data-"&&t!=="aria-");default:return!1}}function xv(t,e,n,i){if(e===null||typeof e>"u"||gv(t,e,n,i))return!0;if(i)return!1;if(n!==null)switch(n.type){case 3:return!e;case 4:return e===!1;case 5:return isNaN(e);case 6:return isNaN(e)||1>e}return!1}function dn(t,e,n,i,r,s,o){this.acceptsBooleans=e===2||e===3||e===4,this.attributeName=i,this.attributeNamespace=r,this.mustUseProperty=n,this.propertyName=t,this.type=e,this.sanitizeURL=s,this.removeEmptyString=o}var Xt={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(t){Xt[t]=new dn(t,0,!1,t,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(t){var e=t[0];Xt[e]=new dn(e,1,!1,t[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(t){Xt[t]=new dn(t,2,!1,t.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(t){Xt[t]=new dn(t,2,!1,t,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(t){Xt[t]=new dn(t,3,!1,t.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(t){Xt[t]=new dn(t,3,!0,t,null,!1,!1)});["capture","download"].forEach(function(t){Xt[t]=new dn(t,4,!1,t,null,!1,!1)});["cols","rows","size","span"].forEach(function(t){Xt[t]=new dn(t,6,!1,t,null,!1,!1)});["rowSpan","start"].forEach(function(t){Xt[t]=new dn(t,5,!1,t.toLowerCase(),null,!1,!1)});var Wf=/[\-:]([a-z])/g;function Xf(t){return t[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(t){var e=t.replace(Wf,Xf);Xt[e]=new dn(e,1,!1,t,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(t){var e=t.replace(Wf,Xf);Xt[e]=new dn(e,1,!1,t,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(t){var e=t.replace(Wf,Xf);Xt[e]=new dn(e,1,!1,t,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(t){Xt[t]=new dn(t,1,!1,t.toLowerCase(),null,!1,!1)});Xt.xlinkHref=new dn("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(t){Xt[t]=new dn(t,1,!1,t.toLowerCase(),null,!0,!0)});function qf(t,e,n,i){var r=Xt.hasOwnProperty(e)?Xt[e]:null;(r!==null?r.type!==0:i||!(2<e.length)||e[0]!=="o"&&e[0]!=="O"||e[1]!=="n"&&e[1]!=="N")&&(xv(e,n,r,i)&&(n=null),i||r===null?mv(e)&&(n===null?t.removeAttribute(e):t.setAttribute(e,""+n)):r.mustUseProperty?t[r.propertyName]=n===null?r.type===3?!1:"":n:(e=r.attributeName,i=r.attributeNamespace,n===null?t.removeAttribute(e):(r=r.type,n=r===3||r===4&&n===!0?"":""+n,i?t.setAttributeNS(i,e,n):t.setAttribute(e,n))))}var ki=pv.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,Sa=Symbol.for("react.element"),gs=Symbol.for("react.portal"),xs=Symbol.for("react.fragment"),$f=Symbol.for("react.strict_mode"),Ku=Symbol.for("react.profiler"),B0=Symbol.for("react.provider"),j0=Symbol.for("react.context"),Kf=Symbol.for("react.forward_ref"),Yu=Symbol.for("react.suspense"),Zu=Symbol.for("react.suspense_list"),Yf=Symbol.for("react.memo"),qi=Symbol.for("react.lazy"),V0=Symbol.for("react.offscreen"),mh=Symbol.iterator;function io(t){return t===null||typeof t!="object"?null:(t=mh&&t[mh]||t["@@iterator"],typeof t=="function"?t:null)}var bt=Object.assign,Bc;function Mo(t){if(Bc===void 0)try{throw Error()}catch(n){var e=n.stack.trim().match(/\n( *(at )?)/);Bc=e&&e[1]||""}return`
`+Bc+t}var jc=!1;function Vc(t,e){if(!t||jc)return"";jc=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(e)if(e=function(){throw Error()},Object.defineProperty(e.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(e,[])}catch(u){var i=u}Reflect.construct(t,[],e)}else{try{e.call()}catch(u){i=u}t.call(e.prototype)}else{try{throw Error()}catch(u){i=u}t()}}catch(u){if(u&&i&&typeof u.stack=="string"){for(var r=u.stack.split(`
`),s=i.stack.split(`
`),o=r.length-1,a=s.length-1;1<=o&&0<=a&&r[o]!==s[a];)a--;for(;1<=o&&0<=a;o--,a--)if(r[o]!==s[a]){if(o!==1||a!==1)do if(o--,a--,0>a||r[o]!==s[a]){var c=`
`+r[o].replace(" at new "," at ");return t.displayName&&c.includes("<anonymous>")&&(c=c.replace("<anonymous>",t.displayName)),c}while(1<=o&&0<=a);break}}}finally{jc=!1,Error.prepareStackTrace=n}return(t=t?t.displayName||t.name:"")?Mo(t):""}function vv(t){switch(t.tag){case 5:return Mo(t.type);case 16:return Mo("Lazy");case 13:return Mo("Suspense");case 19:return Mo("SuspenseList");case 0:case 2:case 15:return t=Vc(t.type,!1),t;case 11:return t=Vc(t.type.render,!1),t;case 1:return t=Vc(t.type,!0),t;default:return""}}function Ju(t){if(t==null)return null;if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t;switch(t){case xs:return"Fragment";case gs:return"Portal";case Ku:return"Profiler";case $f:return"StrictMode";case Yu:return"Suspense";case Zu:return"SuspenseList"}if(typeof t=="object")switch(t.$$typeof){case j0:return(t.displayName||"Context")+".Consumer";case B0:return(t._context.displayName||"Context")+".Provider";case Kf:var e=t.render;return t=t.displayName,t||(t=e.displayName||e.name||"",t=t!==""?"ForwardRef("+t+")":"ForwardRef"),t;case Yf:return e=t.displayName||null,e!==null?e:Ju(t.type)||"Memo";case qi:e=t._payload,t=t._init;try{return Ju(t(e))}catch{}}return null}function _v(t){var e=t.type;switch(t.tag){case 24:return"Cache";case 9:return(e.displayName||"Context")+".Consumer";case 10:return(e._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return t=e.render,t=t.displayName||t.name||"",e.displayName||(t!==""?"ForwardRef("+t+")":"ForwardRef");case 7:return"Fragment";case 5:return e;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return Ju(e);case 8:return e===$f?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e}return null}function dr(t){switch(typeof t){case"boolean":case"number":case"string":case"undefined":return t;case"object":return t;default:return""}}function H0(t){var e=t.type;return(t=t.nodeName)&&t.toLowerCase()==="input"&&(e==="checkbox"||e==="radio")}function yv(t){var e=H0(t)?"checked":"value",n=Object.getOwnPropertyDescriptor(t.constructor.prototype,e),i=""+t[e];if(!t.hasOwnProperty(e)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var r=n.get,s=n.set;return Object.defineProperty(t,e,{configurable:!0,get:function(){return r.call(this)},set:function(o){i=""+o,s.call(this,o)}}),Object.defineProperty(t,e,{enumerable:n.enumerable}),{getValue:function(){return i},setValue:function(o){i=""+o},stopTracking:function(){t._valueTracker=null,delete t[e]}}}}function Ma(t){t._valueTracker||(t._valueTracker=yv(t))}function G0(t){if(!t)return!1;var e=t._valueTracker;if(!e)return!0;var n=e.getValue(),i="";return t&&(i=H0(t)?t.checked?"true":"false":t.value),t=i,t!==n?(e.setValue(t),!0):!1}function Fl(t){if(t=t||(typeof document<"u"?document:void 0),typeof t>"u")return null;try{return t.activeElement||t.body}catch{return t.body}}function Qu(t,e){var n=e.checked;return bt({},e,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??t._wrapperState.initialChecked})}function gh(t,e){var n=e.defaultValue==null?"":e.defaultValue,i=e.checked!=null?e.checked:e.defaultChecked;n=dr(e.value!=null?e.value:n),t._wrapperState={initialChecked:i,initialValue:n,controlled:e.type==="checkbox"||e.type==="radio"?e.checked!=null:e.value!=null}}function W0(t,e){e=e.checked,e!=null&&qf(t,"checked",e,!1)}function ed(t,e){W0(t,e);var n=dr(e.value),i=e.type;if(n!=null)i==="number"?(n===0&&t.value===""||t.value!=n)&&(t.value=""+n):t.value!==""+n&&(t.value=""+n);else if(i==="submit"||i==="reset"){t.removeAttribute("value");return}e.hasOwnProperty("value")?td(t,e.type,n):e.hasOwnProperty("defaultValue")&&td(t,e.type,dr(e.defaultValue)),e.checked==null&&e.defaultChecked!=null&&(t.defaultChecked=!!e.defaultChecked)}function xh(t,e,n){if(e.hasOwnProperty("value")||e.hasOwnProperty("defaultValue")){var i=e.type;if(!(i!=="submit"&&i!=="reset"||e.value!==void 0&&e.value!==null))return;e=""+t._wrapperState.initialValue,n||e===t.value||(t.value=e),t.defaultValue=e}n=t.name,n!==""&&(t.name=""),t.defaultChecked=!!t._wrapperState.initialChecked,n!==""&&(t.name=n)}function td(t,e,n){(e!=="number"||Fl(t.ownerDocument)!==t)&&(n==null?t.defaultValue=""+t._wrapperState.initialValue:t.defaultValue!==""+n&&(t.defaultValue=""+n))}var bo=Array.isArray;function Is(t,e,n,i){if(t=t.options,e){e={};for(var r=0;r<n.length;r++)e["$"+n[r]]=!0;for(n=0;n<t.length;n++)r=e.hasOwnProperty("$"+t[n].value),t[n].selected!==r&&(t[n].selected=r),r&&i&&(t[n].defaultSelected=!0)}else{for(n=""+dr(n),e=null,r=0;r<t.length;r++){if(t[r].value===n){t[r].selected=!0,i&&(t[r].defaultSelected=!0);return}e!==null||t[r].disabled||(e=t[r])}e!==null&&(e.selected=!0)}}function nd(t,e){if(e.dangerouslySetInnerHTML!=null)throw Error(fe(91));return bt({},e,{value:void 0,defaultValue:void 0,children:""+t._wrapperState.initialValue})}function vh(t,e){var n=e.value;if(n==null){if(n=e.children,e=e.defaultValue,n!=null){if(e!=null)throw Error(fe(92));if(bo(n)){if(1<n.length)throw Error(fe(93));n=n[0]}e=n}e==null&&(e=""),n=e}t._wrapperState={initialValue:dr(n)}}function X0(t,e){var n=dr(e.value),i=dr(e.defaultValue);n!=null&&(n=""+n,n!==t.value&&(t.value=n),e.defaultValue==null&&t.defaultValue!==n&&(t.defaultValue=n)),i!=null&&(t.defaultValue=""+i)}function _h(t){var e=t.textContent;e===t._wrapperState.initialValue&&e!==""&&e!==null&&(t.value=e)}function q0(t){switch(t){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function id(t,e){return t==null||t==="http://www.w3.org/1999/xhtml"?q0(e):t==="http://www.w3.org/2000/svg"&&e==="foreignObject"?"http://www.w3.org/1999/xhtml":t}var ba,$0=function(t){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(e,n,i,r){MSApp.execUnsafeLocalFunction(function(){return t(e,n,i,r)})}:t}(function(t,e){if(t.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in t)t.innerHTML=e;else{for(ba=ba||document.createElement("div"),ba.innerHTML="<svg>"+e.valueOf().toString()+"</svg>",e=ba.firstChild;t.firstChild;)t.removeChild(t.firstChild);for(;e.firstChild;)t.appendChild(e.firstChild)}});function Oo(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&n.nodeType===3){n.nodeValue=e;return}}t.textContent=e}var Ao={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Sv=["Webkit","ms","Moz","O"];Object.keys(Ao).forEach(function(t){Sv.forEach(function(e){e=e+t.charAt(0).toUpperCase()+t.substring(1),Ao[e]=Ao[t]})});function K0(t,e,n){return e==null||typeof e=="boolean"||e===""?"":n||typeof e!="number"||e===0||Ao.hasOwnProperty(t)&&Ao[t]?(""+e).trim():e+"px"}function Y0(t,e){t=t.style;for(var n in e)if(e.hasOwnProperty(n)){var i=n.indexOf("--")===0,r=K0(n,e[n],i);n==="float"&&(n="cssFloat"),i?t.setProperty(n,r):t[n]=r}}var Mv=bt({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function rd(t,e){if(e){if(Mv[t]&&(e.children!=null||e.dangerouslySetInnerHTML!=null))throw Error(fe(137,t));if(e.dangerouslySetInnerHTML!=null){if(e.children!=null)throw Error(fe(60));if(typeof e.dangerouslySetInnerHTML!="object"||!("__html"in e.dangerouslySetInnerHTML))throw Error(fe(61))}if(e.style!=null&&typeof e.style!="object")throw Error(fe(62))}}function sd(t,e){if(t.indexOf("-")===-1)return typeof e.is=="string";switch(t){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var od=null;function Zf(t){return t=t.target||t.srcElement||window,t.correspondingUseElement&&(t=t.correspondingUseElement),t.nodeType===3?t.parentNode:t}var ad=null,Ps=null,Ds=null;function yh(t){if(t=ca(t)){if(typeof ad!="function")throw Error(fe(280));var e=t.stateNode;e&&(e=vc(e),ad(t.stateNode,t.type,e))}}function Z0(t){Ps?Ds?Ds.push(t):Ds=[t]:Ps=t}function J0(){if(Ps){var t=Ps,e=Ds;if(Ds=Ps=null,yh(t),e)for(t=0;t<e.length;t++)yh(e[t])}}function Q0(t,e){return t(e)}function eg(){}var Hc=!1;function tg(t,e,n){if(Hc)return t(e,n);Hc=!0;try{return Q0(t,e,n)}finally{Hc=!1,(Ps!==null||Ds!==null)&&(eg(),J0())}}function zo(t,e){var n=t.stateNode;if(n===null)return null;var i=vc(n);if(i===null)return null;n=i[e];e:switch(e){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(i=!i.disabled)||(t=t.type,i=!(t==="button"||t==="input"||t==="select"||t==="textarea")),t=!i;break e;default:t=!1}if(t)return null;if(n&&typeof n!="function")throw Error(fe(231,e,typeof n));return n}var ld=!1;if(Pi)try{var ro={};Object.defineProperty(ro,"passive",{get:function(){ld=!0}}),window.addEventListener("test",ro,ro),window.removeEventListener("test",ro,ro)}catch{ld=!1}function bv(t,e,n,i,r,s,o,a,c){var u=Array.prototype.slice.call(arguments,3);try{e.apply(n,u)}catch(p){this.onError(p)}}var Ro=!1,Nl=null,Ul=!1,cd=null,Ev={onError:function(t){Ro=!0,Nl=t}};function Tv(t,e,n,i,r,s,o,a,c){Ro=!1,Nl=null,bv.apply(Ev,arguments)}function wv(t,e,n,i,r,s,o,a,c){if(Tv.apply(this,arguments),Ro){if(Ro){var u=Nl;Ro=!1,Nl=null}else throw Error(fe(198));Ul||(Ul=!0,cd=u)}}function $r(t){var e=t,n=t;if(t.alternate)for(;e.return;)e=e.return;else{t=e;do e=t,e.flags&4098&&(n=e.return),t=e.return;while(t)}return e.tag===3?n:null}function ng(t){if(t.tag===13){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function Sh(t){if($r(t)!==t)throw Error(fe(188))}function Cv(t){var e=t.alternate;if(!e){if(e=$r(t),e===null)throw Error(fe(188));return e!==t?null:t}for(var n=t,i=e;;){var r=n.return;if(r===null)break;var s=r.alternate;if(s===null){if(i=r.return,i!==null){n=i;continue}break}if(r.child===s.child){for(s=r.child;s;){if(s===n)return Sh(r),t;if(s===i)return Sh(r),e;s=s.sibling}throw Error(fe(188))}if(n.return!==i.return)n=r,i=s;else{for(var o=!1,a=r.child;a;){if(a===n){o=!0,n=r,i=s;break}if(a===i){o=!0,i=r,n=s;break}a=a.sibling}if(!o){for(a=s.child;a;){if(a===n){o=!0,n=s,i=r;break}if(a===i){o=!0,i=s,n=r;break}a=a.sibling}if(!o)throw Error(fe(189))}}if(n.alternate!==i)throw Error(fe(190))}if(n.tag!==3)throw Error(fe(188));return n.stateNode.current===n?t:e}function ig(t){return t=Cv(t),t!==null?rg(t):null}function rg(t){if(t.tag===5||t.tag===6)return t;for(t=t.child;t!==null;){var e=rg(t);if(e!==null)return e;t=t.sibling}return null}var sg=wn.unstable_scheduleCallback,Mh=wn.unstable_cancelCallback,Av=wn.unstable_shouldYield,Rv=wn.unstable_requestPaint,At=wn.unstable_now,Iv=wn.unstable_getCurrentPriorityLevel,Jf=wn.unstable_ImmediatePriority,og=wn.unstable_UserBlockingPriority,kl=wn.unstable_NormalPriority,Pv=wn.unstable_LowPriority,ag=wn.unstable_IdlePriority,hc=null,li=null;function Dv(t){if(li&&typeof li.onCommitFiberRoot=="function")try{li.onCommitFiberRoot(hc,t,void 0,(t.current.flags&128)===128)}catch{}}var qn=Math.clz32?Math.clz32:Nv,Lv=Math.log,Fv=Math.LN2;function Nv(t){return t>>>=0,t===0?32:31-(Lv(t)/Fv|0)|0}var Ea=64,Ta=4194304;function Eo(t){switch(t&-t){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return t&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return t}}function Ol(t,e){var n=t.pendingLanes;if(n===0)return 0;var i=0,r=t.suspendedLanes,s=t.pingedLanes,o=n&268435455;if(o!==0){var a=o&~r;a!==0?i=Eo(a):(s&=o,s!==0&&(i=Eo(s)))}else o=n&~r,o!==0?i=Eo(o):s!==0&&(i=Eo(s));if(i===0)return 0;if(e!==0&&e!==i&&!(e&r)&&(r=i&-i,s=e&-e,r>=s||r===16&&(s&4194240)!==0))return e;if(i&4&&(i|=n&16),e=t.entangledLanes,e!==0)for(t=t.entanglements,e&=i;0<e;)n=31-qn(e),r=1<<n,i|=t[n],e&=~r;return i}function Uv(t,e){switch(t){case 1:case 2:case 4:return e+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function kv(t,e){for(var n=t.suspendedLanes,i=t.pingedLanes,r=t.expirationTimes,s=t.pendingLanes;0<s;){var o=31-qn(s),a=1<<o,c=r[o];c===-1?(!(a&n)||a&i)&&(r[o]=Uv(a,e)):c<=e&&(t.expiredLanes|=a),s&=~a}}function ud(t){return t=t.pendingLanes&-1073741825,t!==0?t:t&1073741824?1073741824:0}function lg(){var t=Ea;return Ea<<=1,!(Ea&4194240)&&(Ea=64),t}function Gc(t){for(var e=[],n=0;31>n;n++)e.push(t);return e}function aa(t,e,n){t.pendingLanes|=e,e!==536870912&&(t.suspendedLanes=0,t.pingedLanes=0),t=t.eventTimes,e=31-qn(e),t[e]=n}function Ov(t,e){var n=t.pendingLanes&~e;t.pendingLanes=e,t.suspendedLanes=0,t.pingedLanes=0,t.expiredLanes&=e,t.mutableReadLanes&=e,t.entangledLanes&=e,e=t.entanglements;var i=t.eventTimes;for(t=t.expirationTimes;0<n;){var r=31-qn(n),s=1<<r;e[r]=0,i[r]=-1,t[r]=-1,n&=~s}}function Qf(t,e){var n=t.entangledLanes|=e;for(t=t.entanglements;n;){var i=31-qn(n),r=1<<i;r&e|t[i]&e&&(t[i]|=e),n&=~r}}var ct=0;function cg(t){return t&=-t,1<t?4<t?t&268435455?16:536870912:4:1}var ug,ep,dg,fg,pg,dd=!1,wa=[],tr=null,nr=null,ir=null,Bo=new Map,jo=new Map,Ki=[],zv="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function bh(t,e){switch(t){case"focusin":case"focusout":tr=null;break;case"dragenter":case"dragleave":nr=null;break;case"mouseover":case"mouseout":ir=null;break;case"pointerover":case"pointerout":Bo.delete(e.pointerId);break;case"gotpointercapture":case"lostpointercapture":jo.delete(e.pointerId)}}function so(t,e,n,i,r,s){return t===null||t.nativeEvent!==s?(t={blockedOn:e,domEventName:n,eventSystemFlags:i,nativeEvent:s,targetContainers:[r]},e!==null&&(e=ca(e),e!==null&&ep(e)),t):(t.eventSystemFlags|=i,e=t.targetContainers,r!==null&&e.indexOf(r)===-1&&e.push(r),t)}function Bv(t,e,n,i,r){switch(e){case"focusin":return tr=so(tr,t,e,n,i,r),!0;case"dragenter":return nr=so(nr,t,e,n,i,r),!0;case"mouseover":return ir=so(ir,t,e,n,i,r),!0;case"pointerover":var s=r.pointerId;return Bo.set(s,so(Bo.get(s)||null,t,e,n,i,r)),!0;case"gotpointercapture":return s=r.pointerId,jo.set(s,so(jo.get(s)||null,t,e,n,i,r)),!0}return!1}function hg(t){var e=Lr(t.target);if(e!==null){var n=$r(e);if(n!==null){if(e=n.tag,e===13){if(e=ng(n),e!==null){t.blockedOn=e,pg(t.priority,function(){dg(n)});return}}else if(e===3&&n.stateNode.current.memoizedState.isDehydrated){t.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}t.blockedOn=null}function gl(t){if(t.blockedOn!==null)return!1;for(var e=t.targetContainers;0<e.length;){var n=fd(t.domEventName,t.eventSystemFlags,e[0],t.nativeEvent);if(n===null){n=t.nativeEvent;var i=new n.constructor(n.type,n);od=i,n.target.dispatchEvent(i),od=null}else return e=ca(n),e!==null&&ep(e),t.blockedOn=n,!1;e.shift()}return!0}function Eh(t,e,n){gl(t)&&n.delete(e)}function jv(){dd=!1,tr!==null&&gl(tr)&&(tr=null),nr!==null&&gl(nr)&&(nr=null),ir!==null&&gl(ir)&&(ir=null),Bo.forEach(Eh),jo.forEach(Eh)}function oo(t,e){t.blockedOn===e&&(t.blockedOn=null,dd||(dd=!0,wn.unstable_scheduleCallback(wn.unstable_NormalPriority,jv)))}function Vo(t){function e(r){return oo(r,t)}if(0<wa.length){oo(wa[0],t);for(var n=1;n<wa.length;n++){var i=wa[n];i.blockedOn===t&&(i.blockedOn=null)}}for(tr!==null&&oo(tr,t),nr!==null&&oo(nr,t),ir!==null&&oo(ir,t),Bo.forEach(e),jo.forEach(e),n=0;n<Ki.length;n++)i=Ki[n],i.blockedOn===t&&(i.blockedOn=null);for(;0<Ki.length&&(n=Ki[0],n.blockedOn===null);)hg(n),n.blockedOn===null&&Ki.shift()}var Ls=ki.ReactCurrentBatchConfig,zl=!0;function Vv(t,e,n,i){var r=ct,s=Ls.transition;Ls.transition=null;try{ct=1,tp(t,e,n,i)}finally{ct=r,Ls.transition=s}}function Hv(t,e,n,i){var r=ct,s=Ls.transition;Ls.transition=null;try{ct=4,tp(t,e,n,i)}finally{ct=r,Ls.transition=s}}function tp(t,e,n,i){if(zl){var r=fd(t,e,n,i);if(r===null)eu(t,e,i,Bl,n),bh(t,i);else if(Bv(r,t,e,n,i))i.stopPropagation();else if(bh(t,i),e&4&&-1<zv.indexOf(t)){for(;r!==null;){var s=ca(r);if(s!==null&&ug(s),s=fd(t,e,n,i),s===null&&eu(t,e,i,Bl,n),s===r)break;r=s}r!==null&&i.stopPropagation()}else eu(t,e,i,null,n)}}var Bl=null;function fd(t,e,n,i){if(Bl=null,t=Zf(i),t=Lr(t),t!==null)if(e=$r(t),e===null)t=null;else if(n=e.tag,n===13){if(t=ng(e),t!==null)return t;t=null}else if(n===3){if(e.stateNode.current.memoizedState.isDehydrated)return e.tag===3?e.stateNode.containerInfo:null;t=null}else e!==t&&(t=null);return Bl=t,null}function mg(t){switch(t){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(Iv()){case Jf:return 1;case og:return 4;case kl:case Pv:return 16;case ag:return 536870912;default:return 16}default:return 16}}var Ji=null,np=null,xl=null;function gg(){if(xl)return xl;var t,e=np,n=e.length,i,r="value"in Ji?Ji.value:Ji.textContent,s=r.length;for(t=0;t<n&&e[t]===r[t];t++);var o=n-t;for(i=1;i<=o&&e[n-i]===r[s-i];i++);return xl=r.slice(t,1<i?1-i:void 0)}function vl(t){var e=t.keyCode;return"charCode"in t?(t=t.charCode,t===0&&e===13&&(t=13)):t=e,t===10&&(t=13),32<=t||t===13?t:0}function Ca(){return!0}function Th(){return!1}function An(t){function e(n,i,r,s,o){this._reactName=n,this._targetInst=r,this.type=i,this.nativeEvent=s,this.target=o,this.currentTarget=null;for(var a in t)t.hasOwnProperty(a)&&(n=t[a],this[a]=n?n(s):s[a]);return this.isDefaultPrevented=(s.defaultPrevented!=null?s.defaultPrevented:s.returnValue===!1)?Ca:Th,this.isPropagationStopped=Th,this}return bt(e.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Ca)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Ca)},persist:function(){},isPersistent:Ca}),e}var Js={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},ip=An(Js),la=bt({},Js,{view:0,detail:0}),Gv=An(la),Wc,Xc,ao,mc=bt({},la,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:rp,button:0,buttons:0,relatedTarget:function(t){return t.relatedTarget===void 0?t.fromElement===t.srcElement?t.toElement:t.fromElement:t.relatedTarget},movementX:function(t){return"movementX"in t?t.movementX:(t!==ao&&(ao&&t.type==="mousemove"?(Wc=t.screenX-ao.screenX,Xc=t.screenY-ao.screenY):Xc=Wc=0,ao=t),Wc)},movementY:function(t){return"movementY"in t?t.movementY:Xc}}),wh=An(mc),Wv=bt({},mc,{dataTransfer:0}),Xv=An(Wv),qv=bt({},la,{relatedTarget:0}),qc=An(qv),$v=bt({},Js,{animationName:0,elapsedTime:0,pseudoElement:0}),Kv=An($v),Yv=bt({},Js,{clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}}),Zv=An(Yv),Jv=bt({},Js,{data:0}),Ch=An(Jv),Qv={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},e_={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},t_={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function n_(t){var e=this.nativeEvent;return e.getModifierState?e.getModifierState(t):(t=t_[t])?!!e[t]:!1}function rp(){return n_}var i_=bt({},la,{key:function(t){if(t.key){var e=Qv[t.key]||t.key;if(e!=="Unidentified")return e}return t.type==="keypress"?(t=vl(t),t===13?"Enter":String.fromCharCode(t)):t.type==="keydown"||t.type==="keyup"?e_[t.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:rp,charCode:function(t){return t.type==="keypress"?vl(t):0},keyCode:function(t){return t.type==="keydown"||t.type==="keyup"?t.keyCode:0},which:function(t){return t.type==="keypress"?vl(t):t.type==="keydown"||t.type==="keyup"?t.keyCode:0}}),r_=An(i_),s_=bt({},mc,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),Ah=An(s_),o_=bt({},la,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:rp}),a_=An(o_),l_=bt({},Js,{propertyName:0,elapsedTime:0,pseudoElement:0}),c_=An(l_),u_=bt({},mc,{deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:0,deltaMode:0}),d_=An(u_),f_=[9,13,27,32],sp=Pi&&"CompositionEvent"in window,Io=null;Pi&&"documentMode"in document&&(Io=document.documentMode);var p_=Pi&&"TextEvent"in window&&!Io,xg=Pi&&(!sp||Io&&8<Io&&11>=Io),Rh=" ",Ih=!1;function vg(t,e){switch(t){case"keyup":return f_.indexOf(e.keyCode)!==-1;case"keydown":return e.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function _g(t){return t=t.detail,typeof t=="object"&&"data"in t?t.data:null}var vs=!1;function h_(t,e){switch(t){case"compositionend":return _g(e);case"keypress":return e.which!==32?null:(Ih=!0,Rh);case"textInput":return t=e.data,t===Rh&&Ih?null:t;default:return null}}function m_(t,e){if(vs)return t==="compositionend"||!sp&&vg(t,e)?(t=gg(),xl=np=Ji=null,vs=!1,t):null;switch(t){case"paste":return null;case"keypress":if(!(e.ctrlKey||e.altKey||e.metaKey)||e.ctrlKey&&e.altKey){if(e.char&&1<e.char.length)return e.char;if(e.which)return String.fromCharCode(e.which)}return null;case"compositionend":return xg&&e.locale!=="ko"?null:e.data;default:return null}}var g_={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Ph(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e==="input"?!!g_[t.type]:e==="textarea"}function yg(t,e,n,i){Z0(i),e=jl(e,"onChange"),0<e.length&&(n=new ip("onChange","change",null,n,i),t.push({event:n,listeners:e}))}var Po=null,Ho=null;function x_(t){Pg(t,0)}function gc(t){var e=Ss(t);if(G0(e))return t}function v_(t,e){if(t==="change")return e}var Sg=!1;if(Pi){var $c;if(Pi){var Kc="oninput"in document;if(!Kc){var Dh=document.createElement("div");Dh.setAttribute("oninput","return;"),Kc=typeof Dh.oninput=="function"}$c=Kc}else $c=!1;Sg=$c&&(!document.documentMode||9<document.documentMode)}function Lh(){Po&&(Po.detachEvent("onpropertychange",Mg),Ho=Po=null)}function Mg(t){if(t.propertyName==="value"&&gc(Ho)){var e=[];yg(e,Ho,t,Zf(t)),tg(x_,e)}}function __(t,e,n){t==="focusin"?(Lh(),Po=e,Ho=n,Po.attachEvent("onpropertychange",Mg)):t==="focusout"&&Lh()}function y_(t){if(t==="selectionchange"||t==="keyup"||t==="keydown")return gc(Ho)}function S_(t,e){if(t==="click")return gc(e)}function M_(t,e){if(t==="input"||t==="change")return gc(e)}function b_(t,e){return t===e&&(t!==0||1/t===1/e)||t!==t&&e!==e}var Yn=typeof Object.is=="function"?Object.is:b_;function Go(t,e){if(Yn(t,e))return!0;if(typeof t!="object"||t===null||typeof e!="object"||e===null)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(i=0;i<n.length;i++){var r=n[i];if(!$u.call(e,r)||!Yn(t[r],e[r]))return!1}return!0}function Fh(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function Nh(t,e){var n=Fh(t);t=0;for(var i;n;){if(n.nodeType===3){if(i=t+n.textContent.length,t<=e&&i>=e)return{node:n,offset:e-t};t=i}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Fh(n)}}function bg(t,e){return t&&e?t===e?!0:t&&t.nodeType===3?!1:e&&e.nodeType===3?bg(t,e.parentNode):"contains"in t?t.contains(e):t.compareDocumentPosition?!!(t.compareDocumentPosition(e)&16):!1:!1}function Eg(){for(var t=window,e=Fl();e instanceof t.HTMLIFrameElement;){try{var n=typeof e.contentWindow.location.href=="string"}catch{n=!1}if(n)t=e.contentWindow;else break;e=Fl(t.document)}return e}function op(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&(e==="input"&&(t.type==="text"||t.type==="search"||t.type==="tel"||t.type==="url"||t.type==="password")||e==="textarea"||t.contentEditable==="true")}function E_(t){var e=Eg(),n=t.focusedElem,i=t.selectionRange;if(e!==n&&n&&n.ownerDocument&&bg(n.ownerDocument.documentElement,n)){if(i!==null&&op(n)){if(e=i.start,t=i.end,t===void 0&&(t=e),"selectionStart"in n)n.selectionStart=e,n.selectionEnd=Math.min(t,n.value.length);else if(t=(e=n.ownerDocument||document)&&e.defaultView||window,t.getSelection){t=t.getSelection();var r=n.textContent.length,s=Math.min(i.start,r);i=i.end===void 0?s:Math.min(i.end,r),!t.extend&&s>i&&(r=i,i=s,s=r),r=Nh(n,s);var o=Nh(n,i);r&&o&&(t.rangeCount!==1||t.anchorNode!==r.node||t.anchorOffset!==r.offset||t.focusNode!==o.node||t.focusOffset!==o.offset)&&(e=e.createRange(),e.setStart(r.node,r.offset),t.removeAllRanges(),s>i?(t.addRange(e),t.extend(o.node,o.offset)):(e.setEnd(o.node,o.offset),t.addRange(e)))}}for(e=[],t=n;t=t.parentNode;)t.nodeType===1&&e.push({element:t,left:t.scrollLeft,top:t.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<e.length;n++)t=e[n],t.element.scrollLeft=t.left,t.element.scrollTop=t.top}}var T_=Pi&&"documentMode"in document&&11>=document.documentMode,_s=null,pd=null,Do=null,hd=!1;function Uh(t,e,n){var i=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;hd||_s==null||_s!==Fl(i)||(i=_s,"selectionStart"in i&&op(i)?i={start:i.selectionStart,end:i.selectionEnd}:(i=(i.ownerDocument&&i.ownerDocument.defaultView||window).getSelection(),i={anchorNode:i.anchorNode,anchorOffset:i.anchorOffset,focusNode:i.focusNode,focusOffset:i.focusOffset}),Do&&Go(Do,i)||(Do=i,i=jl(pd,"onSelect"),0<i.length&&(e=new ip("onSelect","select",null,e,n),t.push({event:e,listeners:i}),e.target=_s)))}function Aa(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n}var ys={animationend:Aa("Animation","AnimationEnd"),animationiteration:Aa("Animation","AnimationIteration"),animationstart:Aa("Animation","AnimationStart"),transitionend:Aa("Transition","TransitionEnd")},Yc={},Tg={};Pi&&(Tg=document.createElement("div").style,"AnimationEvent"in window||(delete ys.animationend.animation,delete ys.animationiteration.animation,delete ys.animationstart.animation),"TransitionEvent"in window||delete ys.transitionend.transition);function xc(t){if(Yc[t])return Yc[t];if(!ys[t])return t;var e=ys[t],n;for(n in e)if(e.hasOwnProperty(n)&&n in Tg)return Yc[t]=e[n];return t}var wg=xc("animationend"),Cg=xc("animationiteration"),Ag=xc("animationstart"),Rg=xc("transitionend"),Ig=new Map,kh="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function mr(t,e){Ig.set(t,e),qr(e,[t])}for(var Zc=0;Zc<kh.length;Zc++){var Jc=kh[Zc],w_=Jc.toLowerCase(),C_=Jc[0].toUpperCase()+Jc.slice(1);mr(w_,"on"+C_)}mr(wg,"onAnimationEnd");mr(Cg,"onAnimationIteration");mr(Ag,"onAnimationStart");mr("dblclick","onDoubleClick");mr("focusin","onFocus");mr("focusout","onBlur");mr(Rg,"onTransitionEnd");Os("onMouseEnter",["mouseout","mouseover"]);Os("onMouseLeave",["mouseout","mouseover"]);Os("onPointerEnter",["pointerout","pointerover"]);Os("onPointerLeave",["pointerout","pointerover"]);qr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));qr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));qr("onBeforeInput",["compositionend","keypress","textInput","paste"]);qr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));qr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));qr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var To="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),A_=new Set("cancel close invalid load scroll toggle".split(" ").concat(To));function Oh(t,e,n){var i=t.type||"unknown-event";t.currentTarget=n,wv(i,e,void 0,t),t.currentTarget=null}function Pg(t,e){e=(e&4)!==0;for(var n=0;n<t.length;n++){var i=t[n],r=i.event;i=i.listeners;e:{var s=void 0;if(e)for(var o=i.length-1;0<=o;o--){var a=i[o],c=a.instance,u=a.currentTarget;if(a=a.listener,c!==s&&r.isPropagationStopped())break e;Oh(r,a,u),s=c}else for(o=0;o<i.length;o++){if(a=i[o],c=a.instance,u=a.currentTarget,a=a.listener,c!==s&&r.isPropagationStopped())break e;Oh(r,a,u),s=c}}}if(Ul)throw t=cd,Ul=!1,cd=null,t}function xt(t,e){var n=e[_d];n===void 0&&(n=e[_d]=new Set);var i=t+"__bubble";n.has(i)||(Dg(e,t,2,!1),n.add(i))}function Qc(t,e,n){var i=0;e&&(i|=4),Dg(n,t,i,e)}var Ra="_reactListening"+Math.random().toString(36).slice(2);function Wo(t){if(!t[Ra]){t[Ra]=!0,z0.forEach(function(n){n!=="selectionchange"&&(A_.has(n)||Qc(n,!1,t),Qc(n,!0,t))});var e=t.nodeType===9?t:t.ownerDocument;e===null||e[Ra]||(e[Ra]=!0,Qc("selectionchange",!1,e))}}function Dg(t,e,n,i){switch(mg(e)){case 1:var r=Vv;break;case 4:r=Hv;break;default:r=tp}n=r.bind(null,e,n,t),r=void 0,!ld||e!=="touchstart"&&e!=="touchmove"&&e!=="wheel"||(r=!0),i?r!==void 0?t.addEventListener(e,n,{capture:!0,passive:r}):t.addEventListener(e,n,!0):r!==void 0?t.addEventListener(e,n,{passive:r}):t.addEventListener(e,n,!1)}function eu(t,e,n,i,r){var s=i;if(!(e&1)&&!(e&2)&&i!==null)e:for(;;){if(i===null)return;var o=i.tag;if(o===3||o===4){var a=i.stateNode.containerInfo;if(a===r||a.nodeType===8&&a.parentNode===r)break;if(o===4)for(o=i.return;o!==null;){var c=o.tag;if((c===3||c===4)&&(c=o.stateNode.containerInfo,c===r||c.nodeType===8&&c.parentNode===r))return;o=o.return}for(;a!==null;){if(o=Lr(a),o===null)return;if(c=o.tag,c===5||c===6){i=s=o;continue e}a=a.parentNode}}i=i.return}tg(function(){var u=s,p=Zf(n),h=[];e:{var f=Ig.get(t);if(f!==void 0){var g=ip,x=t;switch(t){case"keypress":if(vl(n)===0)break e;case"keydown":case"keyup":g=r_;break;case"focusin":x="focus",g=qc;break;case"focusout":x="blur",g=qc;break;case"beforeblur":case"afterblur":g=qc;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":g=wh;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":g=Xv;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":g=a_;break;case wg:case Cg:case Ag:g=Kv;break;case Rg:g=c_;break;case"scroll":g=Gv;break;case"wheel":g=d_;break;case"copy":case"cut":case"paste":g=Zv;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":g=Ah}var M=(e&4)!==0,v=!M&&t==="scroll",d=M?f!==null?f+"Capture":null:f;M=[];for(var m=u,_;m!==null;){_=m;var b=_.stateNode;if(_.tag===5&&b!==null&&(_=b,d!==null&&(b=zo(m,d),b!=null&&M.push(Xo(m,b,_)))),v)break;m=m.return}0<M.length&&(f=new g(f,x,null,n,p),h.push({event:f,listeners:M}))}}if(!(e&7)){e:{if(f=t==="mouseover"||t==="pointerover",g=t==="mouseout"||t==="pointerout",f&&n!==od&&(x=n.relatedTarget||n.fromElement)&&(Lr(x)||x[Di]))break e;if((g||f)&&(f=p.window===p?p:(f=p.ownerDocument)?f.defaultView||f.parentWindow:window,g?(x=n.relatedTarget||n.toElement,g=u,x=x?Lr(x):null,x!==null&&(v=$r(x),x!==v||x.tag!==5&&x.tag!==6)&&(x=null)):(g=null,x=u),g!==x)){if(M=wh,b="onMouseLeave",d="onMouseEnter",m="mouse",(t==="pointerout"||t==="pointerover")&&(M=Ah,b="onPointerLeave",d="onPointerEnter",m="pointer"),v=g==null?f:Ss(g),_=x==null?f:Ss(x),f=new M(b,m+"leave",g,n,p),f.target=v,f.relatedTarget=_,b=null,Lr(p)===u&&(M=new M(d,m+"enter",x,n,p),M.target=_,M.relatedTarget=v,b=M),v=b,g&&x)t:{for(M=g,d=x,m=0,_=M;_;_=Zr(_))m++;for(_=0,b=d;b;b=Zr(b))_++;for(;0<m-_;)M=Zr(M),m--;for(;0<_-m;)d=Zr(d),_--;for(;m--;){if(M===d||d!==null&&M===d.alternate)break t;M=Zr(M),d=Zr(d)}M=null}else M=null;g!==null&&zh(h,f,g,M,!1),x!==null&&v!==null&&zh(h,v,x,M,!0)}}e:{if(f=u?Ss(u):window,g=f.nodeName&&f.nodeName.toLowerCase(),g==="select"||g==="input"&&f.type==="file")var w=v_;else if(Ph(f))if(Sg)w=M_;else{w=y_;var A=__}else(g=f.nodeName)&&g.toLowerCase()==="input"&&(f.type==="checkbox"||f.type==="radio")&&(w=S_);if(w&&(w=w(t,u))){yg(h,w,n,p);break e}A&&A(t,f,u),t==="focusout"&&(A=f._wrapperState)&&A.controlled&&f.type==="number"&&td(f,"number",f.value)}switch(A=u?Ss(u):window,t){case"focusin":(Ph(A)||A.contentEditable==="true")&&(_s=A,pd=u,Do=null);break;case"focusout":Do=pd=_s=null;break;case"mousedown":hd=!0;break;case"contextmenu":case"mouseup":case"dragend":hd=!1,Uh(h,n,p);break;case"selectionchange":if(T_)break;case"keydown":case"keyup":Uh(h,n,p)}var E;if(sp)e:{switch(t){case"compositionstart":var y="onCompositionStart";break e;case"compositionend":y="onCompositionEnd";break e;case"compositionupdate":y="onCompositionUpdate";break e}y=void 0}else vs?vg(t,n)&&(y="onCompositionEnd"):t==="keydown"&&n.keyCode===229&&(y="onCompositionStart");y&&(xg&&n.locale!=="ko"&&(vs||y!=="onCompositionStart"?y==="onCompositionEnd"&&vs&&(E=gg()):(Ji=p,np="value"in Ji?Ji.value:Ji.textContent,vs=!0)),A=jl(u,y),0<A.length&&(y=new Ch(y,t,null,n,p),h.push({event:y,listeners:A}),E?y.data=E:(E=_g(n),E!==null&&(y.data=E)))),(E=p_?h_(t,n):m_(t,n))&&(u=jl(u,"onBeforeInput"),0<u.length&&(p=new Ch("onBeforeInput","beforeinput",null,n,p),h.push({event:p,listeners:u}),p.data=E))}Pg(h,e)})}function Xo(t,e,n){return{instance:t,listener:e,currentTarget:n}}function jl(t,e){for(var n=e+"Capture",i=[];t!==null;){var r=t,s=r.stateNode;r.tag===5&&s!==null&&(r=s,s=zo(t,n),s!=null&&i.unshift(Xo(t,s,r)),s=zo(t,e),s!=null&&i.push(Xo(t,s,r))),t=t.return}return i}function Zr(t){if(t===null)return null;do t=t.return;while(t&&t.tag!==5);return t||null}function zh(t,e,n,i,r){for(var s=e._reactName,o=[];n!==null&&n!==i;){var a=n,c=a.alternate,u=a.stateNode;if(c!==null&&c===i)break;a.tag===5&&u!==null&&(a=u,r?(c=zo(n,s),c!=null&&o.unshift(Xo(n,c,a))):r||(c=zo(n,s),c!=null&&o.push(Xo(n,c,a)))),n=n.return}o.length!==0&&t.push({event:e,listeners:o})}var R_=/\r\n?/g,I_=/\u0000|\uFFFD/g;function Bh(t){return(typeof t=="string"?t:""+t).replace(R_,`
`).replace(I_,"")}function Ia(t,e,n){if(e=Bh(e),Bh(t)!==e&&n)throw Error(fe(425))}function Vl(){}var md=null,gd=null;function xd(t,e){return t==="textarea"||t==="noscript"||typeof e.children=="string"||typeof e.children=="number"||typeof e.dangerouslySetInnerHTML=="object"&&e.dangerouslySetInnerHTML!==null&&e.dangerouslySetInnerHTML.__html!=null}var vd=typeof setTimeout=="function"?setTimeout:void 0,P_=typeof clearTimeout=="function"?clearTimeout:void 0,jh=typeof Promise=="function"?Promise:void 0,D_=typeof queueMicrotask=="function"?queueMicrotask:typeof jh<"u"?function(t){return jh.resolve(null).then(t).catch(L_)}:vd;function L_(t){setTimeout(function(){throw t})}function tu(t,e){var n=e,i=0;do{var r=n.nextSibling;if(t.removeChild(n),r&&r.nodeType===8)if(n=r.data,n==="/$"){if(i===0){t.removeChild(r),Vo(e);return}i--}else n!=="$"&&n!=="$?"&&n!=="$!"||i++;n=r}while(n);Vo(e)}function rr(t){for(;t!=null;t=t.nextSibling){var e=t.nodeType;if(e===1||e===3)break;if(e===8){if(e=t.data,e==="$"||e==="$!"||e==="$?")break;if(e==="/$")return null}}return t}function Vh(t){t=t.previousSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="$"||n==="$!"||n==="$?"){if(e===0)return t;e--}else n==="/$"&&e++}t=t.previousSibling}return null}var Qs=Math.random().toString(36).slice(2),ii="__reactFiber$"+Qs,qo="__reactProps$"+Qs,Di="__reactContainer$"+Qs,_d="__reactEvents$"+Qs,F_="__reactListeners$"+Qs,N_="__reactHandles$"+Qs;function Lr(t){var e=t[ii];if(e)return e;for(var n=t.parentNode;n;){if(e=n[Di]||n[ii]){if(n=e.alternate,e.child!==null||n!==null&&n.child!==null)for(t=Vh(t);t!==null;){if(n=t[ii])return n;t=Vh(t)}return e}t=n,n=t.parentNode}return null}function ca(t){return t=t[ii]||t[Di],!t||t.tag!==5&&t.tag!==6&&t.tag!==13&&t.tag!==3?null:t}function Ss(t){if(t.tag===5||t.tag===6)return t.stateNode;throw Error(fe(33))}function vc(t){return t[qo]||null}var yd=[],Ms=-1;function gr(t){return{current:t}}function vt(t){0>Ms||(t.current=yd[Ms],yd[Ms]=null,Ms--)}function gt(t,e){Ms++,yd[Ms]=t.current,t.current=e}var fr={},nn=gr(fr),mn=gr(!1),jr=fr;function zs(t,e){var n=t.type.contextTypes;if(!n)return fr;var i=t.stateNode;if(i&&i.__reactInternalMemoizedUnmaskedChildContext===e)return i.__reactInternalMemoizedMaskedChildContext;var r={},s;for(s in n)r[s]=e[s];return i&&(t=t.stateNode,t.__reactInternalMemoizedUnmaskedChildContext=e,t.__reactInternalMemoizedMaskedChildContext=r),r}function gn(t){return t=t.childContextTypes,t!=null}function Hl(){vt(mn),vt(nn)}function Hh(t,e,n){if(nn.current!==fr)throw Error(fe(168));gt(nn,e),gt(mn,n)}function Lg(t,e,n){var i=t.stateNode;if(e=e.childContextTypes,typeof i.getChildContext!="function")return n;i=i.getChildContext();for(var r in i)if(!(r in e))throw Error(fe(108,_v(t)||"Unknown",r));return bt({},n,i)}function Gl(t){return t=(t=t.stateNode)&&t.__reactInternalMemoizedMergedChildContext||fr,jr=nn.current,gt(nn,t),gt(mn,mn.current),!0}function Gh(t,e,n){var i=t.stateNode;if(!i)throw Error(fe(169));n?(t=Lg(t,e,jr),i.__reactInternalMemoizedMergedChildContext=t,vt(mn),vt(nn),gt(nn,t)):vt(mn),gt(mn,n)}var Ei=null,_c=!1,nu=!1;function Fg(t){Ei===null?Ei=[t]:Ei.push(t)}function U_(t){_c=!0,Fg(t)}function xr(){if(!nu&&Ei!==null){nu=!0;var t=0,e=ct;try{var n=Ei;for(ct=1;t<n.length;t++){var i=n[t];do i=i(!0);while(i!==null)}Ei=null,_c=!1}catch(r){throw Ei!==null&&(Ei=Ei.slice(t+1)),sg(Jf,xr),r}finally{ct=e,nu=!1}}return null}var bs=[],Es=0,Wl=null,Xl=0,Pn=[],Dn=0,Vr=null,Ti=1,wi="";function Ar(t,e){bs[Es++]=Xl,bs[Es++]=Wl,Wl=t,Xl=e}function Ng(t,e,n){Pn[Dn++]=Ti,Pn[Dn++]=wi,Pn[Dn++]=Vr,Vr=t;var i=Ti;t=wi;var r=32-qn(i)-1;i&=~(1<<r),n+=1;var s=32-qn(e)+r;if(30<s){var o=r-r%5;s=(i&(1<<o)-1).toString(32),i>>=o,r-=o,Ti=1<<32-qn(e)+r|n<<r|i,wi=s+t}else Ti=1<<s|n<<r|i,wi=t}function ap(t){t.return!==null&&(Ar(t,1),Ng(t,1,0))}function lp(t){for(;t===Wl;)Wl=bs[--Es],bs[Es]=null,Xl=bs[--Es],bs[Es]=null;for(;t===Vr;)Vr=Pn[--Dn],Pn[Dn]=null,wi=Pn[--Dn],Pn[Dn]=null,Ti=Pn[--Dn],Pn[Dn]=null}var Tn=null,En=null,yt=!1,Wn=null;function Ug(t,e){var n=Fn(5,null,null,0);n.elementType="DELETED",n.stateNode=e,n.return=t,e=t.deletions,e===null?(t.deletions=[n],t.flags|=16):e.push(n)}function Wh(t,e){switch(t.tag){case 5:var n=t.type;return e=e.nodeType!==1||n.toLowerCase()!==e.nodeName.toLowerCase()?null:e,e!==null?(t.stateNode=e,Tn=t,En=rr(e.firstChild),!0):!1;case 6:return e=t.pendingProps===""||e.nodeType!==3?null:e,e!==null?(t.stateNode=e,Tn=t,En=null,!0):!1;case 13:return e=e.nodeType!==8?null:e,e!==null?(n=Vr!==null?{id:Ti,overflow:wi}:null,t.memoizedState={dehydrated:e,treeContext:n,retryLane:1073741824},n=Fn(18,null,null,0),n.stateNode=e,n.return=t,t.child=n,Tn=t,En=null,!0):!1;default:return!1}}function Sd(t){return(t.mode&1)!==0&&(t.flags&128)===0}function Md(t){if(yt){var e=En;if(e){var n=e;if(!Wh(t,e)){if(Sd(t))throw Error(fe(418));e=rr(n.nextSibling);var i=Tn;e&&Wh(t,e)?Ug(i,n):(t.flags=t.flags&-4097|2,yt=!1,Tn=t)}}else{if(Sd(t))throw Error(fe(418));t.flags=t.flags&-4097|2,yt=!1,Tn=t}}}function Xh(t){for(t=t.return;t!==null&&t.tag!==5&&t.tag!==3&&t.tag!==13;)t=t.return;Tn=t}function Pa(t){if(t!==Tn)return!1;if(!yt)return Xh(t),yt=!0,!1;var e;if((e=t.tag!==3)&&!(e=t.tag!==5)&&(e=t.type,e=e!=="head"&&e!=="body"&&!xd(t.type,t.memoizedProps)),e&&(e=En)){if(Sd(t))throw kg(),Error(fe(418));for(;e;)Ug(t,e),e=rr(e.nextSibling)}if(Xh(t),t.tag===13){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(fe(317));e:{for(t=t.nextSibling,e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="/$"){if(e===0){En=rr(t.nextSibling);break e}e--}else n!=="$"&&n!=="$!"&&n!=="$?"||e++}t=t.nextSibling}En=null}}else En=Tn?rr(t.stateNode.nextSibling):null;return!0}function kg(){for(var t=En;t;)t=rr(t.nextSibling)}function Bs(){En=Tn=null,yt=!1}function cp(t){Wn===null?Wn=[t]:Wn.push(t)}var k_=ki.ReactCurrentBatchConfig;function lo(t,e,n){if(t=n.ref,t!==null&&typeof t!="function"&&typeof t!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(fe(309));var i=n.stateNode}if(!i)throw Error(fe(147,t));var r=i,s=""+t;return e!==null&&e.ref!==null&&typeof e.ref=="function"&&e.ref._stringRef===s?e.ref:(e=function(o){var a=r.refs;o===null?delete a[s]:a[s]=o},e._stringRef=s,e)}if(typeof t!="string")throw Error(fe(284));if(!n._owner)throw Error(fe(290,t))}return t}function Da(t,e){throw t=Object.prototype.toString.call(e),Error(fe(31,t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t))}function qh(t){var e=t._init;return e(t._payload)}function Og(t){function e(d,m){if(t){var _=d.deletions;_===null?(d.deletions=[m],d.flags|=16):_.push(m)}}function n(d,m){if(!t)return null;for(;m!==null;)e(d,m),m=m.sibling;return null}function i(d,m){for(d=new Map;m!==null;)m.key!==null?d.set(m.key,m):d.set(m.index,m),m=m.sibling;return d}function r(d,m){return d=lr(d,m),d.index=0,d.sibling=null,d}function s(d,m,_){return d.index=_,t?(_=d.alternate,_!==null?(_=_.index,_<m?(d.flags|=2,m):_):(d.flags|=2,m)):(d.flags|=1048576,m)}function o(d){return t&&d.alternate===null&&(d.flags|=2),d}function a(d,m,_,b){return m===null||m.tag!==6?(m=cu(_,d.mode,b),m.return=d,m):(m=r(m,_),m.return=d,m)}function c(d,m,_,b){var w=_.type;return w===xs?p(d,m,_.props.children,b,_.key):m!==null&&(m.elementType===w||typeof w=="object"&&w!==null&&w.$$typeof===qi&&qh(w)===m.type)?(b=r(m,_.props),b.ref=lo(d,m,_),b.return=d,b):(b=Tl(_.type,_.key,_.props,null,d.mode,b),b.ref=lo(d,m,_),b.return=d,b)}function u(d,m,_,b){return m===null||m.tag!==4||m.stateNode.containerInfo!==_.containerInfo||m.stateNode.implementation!==_.implementation?(m=uu(_,d.mode,b),m.return=d,m):(m=r(m,_.children||[]),m.return=d,m)}function p(d,m,_,b,w){return m===null||m.tag!==7?(m=zr(_,d.mode,b,w),m.return=d,m):(m=r(m,_),m.return=d,m)}function h(d,m,_){if(typeof m=="string"&&m!==""||typeof m=="number")return m=cu(""+m,d.mode,_),m.return=d,m;if(typeof m=="object"&&m!==null){switch(m.$$typeof){case Sa:return _=Tl(m.type,m.key,m.props,null,d.mode,_),_.ref=lo(d,null,m),_.return=d,_;case gs:return m=uu(m,d.mode,_),m.return=d,m;case qi:var b=m._init;return h(d,b(m._payload),_)}if(bo(m)||io(m))return m=zr(m,d.mode,_,null),m.return=d,m;Da(d,m)}return null}function f(d,m,_,b){var w=m!==null?m.key:null;if(typeof _=="string"&&_!==""||typeof _=="number")return w!==null?null:a(d,m,""+_,b);if(typeof _=="object"&&_!==null){switch(_.$$typeof){case Sa:return _.key===w?c(d,m,_,b):null;case gs:return _.key===w?u(d,m,_,b):null;case qi:return w=_._init,f(d,m,w(_._payload),b)}if(bo(_)||io(_))return w!==null?null:p(d,m,_,b,null);Da(d,_)}return null}function g(d,m,_,b,w){if(typeof b=="string"&&b!==""||typeof b=="number")return d=d.get(_)||null,a(m,d,""+b,w);if(typeof b=="object"&&b!==null){switch(b.$$typeof){case Sa:return d=d.get(b.key===null?_:b.key)||null,c(m,d,b,w);case gs:return d=d.get(b.key===null?_:b.key)||null,u(m,d,b,w);case qi:var A=b._init;return g(d,m,_,A(b._payload),w)}if(bo(b)||io(b))return d=d.get(_)||null,p(m,d,b,w,null);Da(m,b)}return null}function x(d,m,_,b){for(var w=null,A=null,E=m,y=m=0,C=null;E!==null&&y<_.length;y++){E.index>y?(C=E,E=null):C=E.sibling;var P=f(d,E,_[y],b);if(P===null){E===null&&(E=C);break}t&&E&&P.alternate===null&&e(d,E),m=s(P,m,y),A===null?w=P:A.sibling=P,A=P,E=C}if(y===_.length)return n(d,E),yt&&Ar(d,y),w;if(E===null){for(;y<_.length;y++)E=h(d,_[y],b),E!==null&&(m=s(E,m,y),A===null?w=E:A.sibling=E,A=E);return yt&&Ar(d,y),w}for(E=i(d,E);y<_.length;y++)C=g(E,d,y,_[y],b),C!==null&&(t&&C.alternate!==null&&E.delete(C.key===null?y:C.key),m=s(C,m,y),A===null?w=C:A.sibling=C,A=C);return t&&E.forEach(function(I){return e(d,I)}),yt&&Ar(d,y),w}function M(d,m,_,b){var w=io(_);if(typeof w!="function")throw Error(fe(150));if(_=w.call(_),_==null)throw Error(fe(151));for(var A=w=null,E=m,y=m=0,C=null,P=_.next();E!==null&&!P.done;y++,P=_.next()){E.index>y?(C=E,E=null):C=E.sibling;var I=f(d,E,P.value,b);if(I===null){E===null&&(E=C);break}t&&E&&I.alternate===null&&e(d,E),m=s(I,m,y),A===null?w=I:A.sibling=I,A=I,E=C}if(P.done)return n(d,E),yt&&Ar(d,y),w;if(E===null){for(;!P.done;y++,P=_.next())P=h(d,P.value,b),P!==null&&(m=s(P,m,y),A===null?w=P:A.sibling=P,A=P);return yt&&Ar(d,y),w}for(E=i(d,E);!P.done;y++,P=_.next())P=g(E,d,y,P.value,b),P!==null&&(t&&P.alternate!==null&&E.delete(P.key===null?y:P.key),m=s(P,m,y),A===null?w=P:A.sibling=P,A=P);return t&&E.forEach(function(F){return e(d,F)}),yt&&Ar(d,y),w}function v(d,m,_,b){if(typeof _=="object"&&_!==null&&_.type===xs&&_.key===null&&(_=_.props.children),typeof _=="object"&&_!==null){switch(_.$$typeof){case Sa:e:{for(var w=_.key,A=m;A!==null;){if(A.key===w){if(w=_.type,w===xs){if(A.tag===7){n(d,A.sibling),m=r(A,_.props.children),m.return=d,d=m;break e}}else if(A.elementType===w||typeof w=="object"&&w!==null&&w.$$typeof===qi&&qh(w)===A.type){n(d,A.sibling),m=r(A,_.props),m.ref=lo(d,A,_),m.return=d,d=m;break e}n(d,A);break}else e(d,A);A=A.sibling}_.type===xs?(m=zr(_.props.children,d.mode,b,_.key),m.return=d,d=m):(b=Tl(_.type,_.key,_.props,null,d.mode,b),b.ref=lo(d,m,_),b.return=d,d=b)}return o(d);case gs:e:{for(A=_.key;m!==null;){if(m.key===A)if(m.tag===4&&m.stateNode.containerInfo===_.containerInfo&&m.stateNode.implementation===_.implementation){n(d,m.sibling),m=r(m,_.children||[]),m.return=d,d=m;break e}else{n(d,m);break}else e(d,m);m=m.sibling}m=uu(_,d.mode,b),m.return=d,d=m}return o(d);case qi:return A=_._init,v(d,m,A(_._payload),b)}if(bo(_))return x(d,m,_,b);if(io(_))return M(d,m,_,b);Da(d,_)}return typeof _=="string"&&_!==""||typeof _=="number"?(_=""+_,m!==null&&m.tag===6?(n(d,m.sibling),m=r(m,_),m.return=d,d=m):(n(d,m),m=cu(_,d.mode,b),m.return=d,d=m),o(d)):n(d,m)}return v}var js=Og(!0),zg=Og(!1),ql=gr(null),$l=null,Ts=null,up=null;function dp(){up=Ts=$l=null}function fp(t){var e=ql.current;vt(ql),t._currentValue=e}function bd(t,e,n){for(;t!==null;){var i=t.alternate;if((t.childLanes&e)!==e?(t.childLanes|=e,i!==null&&(i.childLanes|=e)):i!==null&&(i.childLanes&e)!==e&&(i.childLanes|=e),t===n)break;t=t.return}}function Fs(t,e){$l=t,up=Ts=null,t=t.dependencies,t!==null&&t.firstContext!==null&&(t.lanes&e&&(hn=!0),t.firstContext=null)}function Un(t){var e=t._currentValue;if(up!==t)if(t={context:t,memoizedValue:e,next:null},Ts===null){if($l===null)throw Error(fe(308));Ts=t,$l.dependencies={lanes:0,firstContext:t}}else Ts=Ts.next=t;return e}var Fr=null;function pp(t){Fr===null?Fr=[t]:Fr.push(t)}function Bg(t,e,n,i){var r=e.interleaved;return r===null?(n.next=n,pp(e)):(n.next=r.next,r.next=n),e.interleaved=n,Li(t,i)}function Li(t,e){t.lanes|=e;var n=t.alternate;for(n!==null&&(n.lanes|=e),n=t,t=t.return;t!==null;)t.childLanes|=e,n=t.alternate,n!==null&&(n.childLanes|=e),n=t,t=t.return;return n.tag===3?n.stateNode:null}var $i=!1;function hp(t){t.updateQueue={baseState:t.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function jg(t,e){t=t.updateQueue,e.updateQueue===t&&(e.updateQueue={baseState:t.baseState,firstBaseUpdate:t.firstBaseUpdate,lastBaseUpdate:t.lastBaseUpdate,shared:t.shared,effects:t.effects})}function Ai(t,e){return{eventTime:t,lane:e,tag:0,payload:null,callback:null,next:null}}function sr(t,e,n){var i=t.updateQueue;if(i===null)return null;if(i=i.shared,st&2){var r=i.pending;return r===null?e.next=e:(e.next=r.next,r.next=e),i.pending=e,Li(t,n)}return r=i.interleaved,r===null?(e.next=e,pp(i)):(e.next=r.next,r.next=e),i.interleaved=e,Li(t,n)}function _l(t,e,n){if(e=e.updateQueue,e!==null&&(e=e.shared,(n&4194240)!==0)){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,Qf(t,n)}}function $h(t,e){var n=t.updateQueue,i=t.alternate;if(i!==null&&(i=i.updateQueue,n===i)){var r=null,s=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};s===null?r=s=o:s=s.next=o,n=n.next}while(n!==null);s===null?r=s=e:s=s.next=e}else r=s=e;n={baseState:i.baseState,firstBaseUpdate:r,lastBaseUpdate:s,shared:i.shared,effects:i.effects},t.updateQueue=n;return}t=n.lastBaseUpdate,t===null?n.firstBaseUpdate=e:t.next=e,n.lastBaseUpdate=e}function Kl(t,e,n,i){var r=t.updateQueue;$i=!1;var s=r.firstBaseUpdate,o=r.lastBaseUpdate,a=r.shared.pending;if(a!==null){r.shared.pending=null;var c=a,u=c.next;c.next=null,o===null?s=u:o.next=u,o=c;var p=t.alternate;p!==null&&(p=p.updateQueue,a=p.lastBaseUpdate,a!==o&&(a===null?p.firstBaseUpdate=u:a.next=u,p.lastBaseUpdate=c))}if(s!==null){var h=r.baseState;o=0,p=u=c=null,a=s;do{var f=a.lane,g=a.eventTime;if((i&f)===f){p!==null&&(p=p.next={eventTime:g,lane:0,tag:a.tag,payload:a.payload,callback:a.callback,next:null});e:{var x=t,M=a;switch(f=e,g=n,M.tag){case 1:if(x=M.payload,typeof x=="function"){h=x.call(g,h,f);break e}h=x;break e;case 3:x.flags=x.flags&-65537|128;case 0:if(x=M.payload,f=typeof x=="function"?x.call(g,h,f):x,f==null)break e;h=bt({},h,f);break e;case 2:$i=!0}}a.callback!==null&&a.lane!==0&&(t.flags|=64,f=r.effects,f===null?r.effects=[a]:f.push(a))}else g={eventTime:g,lane:f,tag:a.tag,payload:a.payload,callback:a.callback,next:null},p===null?(u=p=g,c=h):p=p.next=g,o|=f;if(a=a.next,a===null){if(a=r.shared.pending,a===null)break;f=a,a=f.next,f.next=null,r.lastBaseUpdate=f,r.shared.pending=null}}while(!0);if(p===null&&(c=h),r.baseState=c,r.firstBaseUpdate=u,r.lastBaseUpdate=p,e=r.shared.interleaved,e!==null){r=e;do o|=r.lane,r=r.next;while(r!==e)}else s===null&&(r.shared.lanes=0);Gr|=o,t.lanes=o,t.memoizedState=h}}function Kh(t,e,n){if(t=e.effects,e.effects=null,t!==null)for(e=0;e<t.length;e++){var i=t[e],r=i.callback;if(r!==null){if(i.callback=null,i=n,typeof r!="function")throw Error(fe(191,r));r.call(i)}}}var ua={},ci=gr(ua),$o=gr(ua),Ko=gr(ua);function Nr(t){if(t===ua)throw Error(fe(174));return t}function mp(t,e){switch(gt(Ko,e),gt($o,t),gt(ci,ua),t=e.nodeType,t){case 9:case 11:e=(e=e.documentElement)?e.namespaceURI:id(null,"");break;default:t=t===8?e.parentNode:e,e=t.namespaceURI||null,t=t.tagName,e=id(e,t)}vt(ci),gt(ci,e)}function Vs(){vt(ci),vt($o),vt(Ko)}function Vg(t){Nr(Ko.current);var e=Nr(ci.current),n=id(e,t.type);e!==n&&(gt($o,t),gt(ci,n))}function gp(t){$o.current===t&&(vt(ci),vt($o))}var St=gr(0);function Yl(t){for(var e=t;e!==null;){if(e.tag===13){var n=e.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return e}else if(e.tag===19&&e.memoizedProps.revealOrder!==void 0){if(e.flags&128)return e}else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return null;e=e.return}e.sibling.return=e.return,e=e.sibling}return null}var iu=[];function xp(){for(var t=0;t<iu.length;t++)iu[t]._workInProgressVersionPrimary=null;iu.length=0}var yl=ki.ReactCurrentDispatcher,ru=ki.ReactCurrentBatchConfig,Hr=0,Mt=null,Pt=null,kt=null,Zl=!1,Lo=!1,Yo=0,O_=0;function $t(){throw Error(fe(321))}function vp(t,e){if(e===null)return!1;for(var n=0;n<e.length&&n<t.length;n++)if(!Yn(t[n],e[n]))return!1;return!0}function _p(t,e,n,i,r,s){if(Hr=s,Mt=e,e.memoizedState=null,e.updateQueue=null,e.lanes=0,yl.current=t===null||t.memoizedState===null?V_:H_,t=n(i,r),Lo){s=0;do{if(Lo=!1,Yo=0,25<=s)throw Error(fe(301));s+=1,kt=Pt=null,e.updateQueue=null,yl.current=G_,t=n(i,r)}while(Lo)}if(yl.current=Jl,e=Pt!==null&&Pt.next!==null,Hr=0,kt=Pt=Mt=null,Zl=!1,e)throw Error(fe(300));return t}function yp(){var t=Yo!==0;return Yo=0,t}function ti(){var t={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return kt===null?Mt.memoizedState=kt=t:kt=kt.next=t,kt}function kn(){if(Pt===null){var t=Mt.alternate;t=t!==null?t.memoizedState:null}else t=Pt.next;var e=kt===null?Mt.memoizedState:kt.next;if(e!==null)kt=e,Pt=t;else{if(t===null)throw Error(fe(310));Pt=t,t={memoizedState:Pt.memoizedState,baseState:Pt.baseState,baseQueue:Pt.baseQueue,queue:Pt.queue,next:null},kt===null?Mt.memoizedState=kt=t:kt=kt.next=t}return kt}function Zo(t,e){return typeof e=="function"?e(t):e}function su(t){var e=kn(),n=e.queue;if(n===null)throw Error(fe(311));n.lastRenderedReducer=t;var i=Pt,r=i.baseQueue,s=n.pending;if(s!==null){if(r!==null){var o=r.next;r.next=s.next,s.next=o}i.baseQueue=r=s,n.pending=null}if(r!==null){s=r.next,i=i.baseState;var a=o=null,c=null,u=s;do{var p=u.lane;if((Hr&p)===p)c!==null&&(c=c.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),i=u.hasEagerState?u.eagerState:t(i,u.action);else{var h={lane:p,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};c===null?(a=c=h,o=i):c=c.next=h,Mt.lanes|=p,Gr|=p}u=u.next}while(u!==null&&u!==s);c===null?o=i:c.next=a,Yn(i,e.memoizedState)||(hn=!0),e.memoizedState=i,e.baseState=o,e.baseQueue=c,n.lastRenderedState=i}if(t=n.interleaved,t!==null){r=t;do s=r.lane,Mt.lanes|=s,Gr|=s,r=r.next;while(r!==t)}else r===null&&(n.lanes=0);return[e.memoizedState,n.dispatch]}function ou(t){var e=kn(),n=e.queue;if(n===null)throw Error(fe(311));n.lastRenderedReducer=t;var i=n.dispatch,r=n.pending,s=e.memoizedState;if(r!==null){n.pending=null;var o=r=r.next;do s=t(s,o.action),o=o.next;while(o!==r);Yn(s,e.memoizedState)||(hn=!0),e.memoizedState=s,e.baseQueue===null&&(e.baseState=s),n.lastRenderedState=s}return[s,i]}function Hg(){}function Gg(t,e){var n=Mt,i=kn(),r=e(),s=!Yn(i.memoizedState,r);if(s&&(i.memoizedState=r,hn=!0),i=i.queue,Sp(qg.bind(null,n,i,t),[t]),i.getSnapshot!==e||s||kt!==null&&kt.memoizedState.tag&1){if(n.flags|=2048,Jo(9,Xg.bind(null,n,i,r,e),void 0,null),Ot===null)throw Error(fe(349));Hr&30||Wg(n,e,r)}return r}function Wg(t,e,n){t.flags|=16384,t={getSnapshot:e,value:n},e=Mt.updateQueue,e===null?(e={lastEffect:null,stores:null},Mt.updateQueue=e,e.stores=[t]):(n=e.stores,n===null?e.stores=[t]:n.push(t))}function Xg(t,e,n,i){e.value=n,e.getSnapshot=i,$g(e)&&Kg(t)}function qg(t,e,n){return n(function(){$g(e)&&Kg(t)})}function $g(t){var e=t.getSnapshot;t=t.value;try{var n=e();return!Yn(t,n)}catch{return!0}}function Kg(t){var e=Li(t,1);e!==null&&$n(e,t,1,-1)}function Yh(t){var e=ti();return typeof t=="function"&&(t=t()),e.memoizedState=e.baseState=t,t={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:Zo,lastRenderedState:t},e.queue=t,t=t.dispatch=j_.bind(null,Mt,t),[e.memoizedState,t]}function Jo(t,e,n,i){return t={tag:t,create:e,destroy:n,deps:i,next:null},e=Mt.updateQueue,e===null?(e={lastEffect:null,stores:null},Mt.updateQueue=e,e.lastEffect=t.next=t):(n=e.lastEffect,n===null?e.lastEffect=t.next=t:(i=n.next,n.next=t,t.next=i,e.lastEffect=t)),t}function Yg(){return kn().memoizedState}function Sl(t,e,n,i){var r=ti();Mt.flags|=t,r.memoizedState=Jo(1|e,n,void 0,i===void 0?null:i)}function yc(t,e,n,i){var r=kn();i=i===void 0?null:i;var s=void 0;if(Pt!==null){var o=Pt.memoizedState;if(s=o.destroy,i!==null&&vp(i,o.deps)){r.memoizedState=Jo(e,n,s,i);return}}Mt.flags|=t,r.memoizedState=Jo(1|e,n,s,i)}function Zh(t,e){return Sl(8390656,8,t,e)}function Sp(t,e){return yc(2048,8,t,e)}function Zg(t,e){return yc(4,2,t,e)}function Jg(t,e){return yc(4,4,t,e)}function Qg(t,e){if(typeof e=="function")return t=t(),e(t),function(){e(null)};if(e!=null)return t=t(),e.current=t,function(){e.current=null}}function e1(t,e,n){return n=n!=null?n.concat([t]):null,yc(4,4,Qg.bind(null,e,t),n)}function Mp(){}function t1(t,e){var n=kn();e=e===void 0?null:e;var i=n.memoizedState;return i!==null&&e!==null&&vp(e,i[1])?i[0]:(n.memoizedState=[t,e],t)}function n1(t,e){var n=kn();e=e===void 0?null:e;var i=n.memoizedState;return i!==null&&e!==null&&vp(e,i[1])?i[0]:(t=t(),n.memoizedState=[t,e],t)}function i1(t,e,n){return Hr&21?(Yn(n,e)||(n=lg(),Mt.lanes|=n,Gr|=n,t.baseState=!0),e):(t.baseState&&(t.baseState=!1,hn=!0),t.memoizedState=n)}function z_(t,e){var n=ct;ct=n!==0&&4>n?n:4,t(!0);var i=ru.transition;ru.transition={};try{t(!1),e()}finally{ct=n,ru.transition=i}}function r1(){return kn().memoizedState}function B_(t,e,n){var i=ar(t);if(n={lane:i,action:n,hasEagerState:!1,eagerState:null,next:null},s1(t))o1(e,n);else if(n=Bg(t,e,n,i),n!==null){var r=cn();$n(n,t,i,r),a1(n,e,i)}}function j_(t,e,n){var i=ar(t),r={lane:i,action:n,hasEagerState:!1,eagerState:null,next:null};if(s1(t))o1(e,r);else{var s=t.alternate;if(t.lanes===0&&(s===null||s.lanes===0)&&(s=e.lastRenderedReducer,s!==null))try{var o=e.lastRenderedState,a=s(o,n);if(r.hasEagerState=!0,r.eagerState=a,Yn(a,o)){var c=e.interleaved;c===null?(r.next=r,pp(e)):(r.next=c.next,c.next=r),e.interleaved=r;return}}catch{}finally{}n=Bg(t,e,r,i),n!==null&&(r=cn(),$n(n,t,i,r),a1(n,e,i))}}function s1(t){var e=t.alternate;return t===Mt||e!==null&&e===Mt}function o1(t,e){Lo=Zl=!0;var n=t.pending;n===null?e.next=e:(e.next=n.next,n.next=e),t.pending=e}function a1(t,e,n){if(n&4194240){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,Qf(t,n)}}var Jl={readContext:Un,useCallback:$t,useContext:$t,useEffect:$t,useImperativeHandle:$t,useInsertionEffect:$t,useLayoutEffect:$t,useMemo:$t,useReducer:$t,useRef:$t,useState:$t,useDebugValue:$t,useDeferredValue:$t,useTransition:$t,useMutableSource:$t,useSyncExternalStore:$t,useId:$t,unstable_isNewReconciler:!1},V_={readContext:Un,useCallback:function(t,e){return ti().memoizedState=[t,e===void 0?null:e],t},useContext:Un,useEffect:Zh,useImperativeHandle:function(t,e,n){return n=n!=null?n.concat([t]):null,Sl(4194308,4,Qg.bind(null,e,t),n)},useLayoutEffect:function(t,e){return Sl(4194308,4,t,e)},useInsertionEffect:function(t,e){return Sl(4,2,t,e)},useMemo:function(t,e){var n=ti();return e=e===void 0?null:e,t=t(),n.memoizedState=[t,e],t},useReducer:function(t,e,n){var i=ti();return e=n!==void 0?n(e):e,i.memoizedState=i.baseState=e,t={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:t,lastRenderedState:e},i.queue=t,t=t.dispatch=B_.bind(null,Mt,t),[i.memoizedState,t]},useRef:function(t){var e=ti();return t={current:t},e.memoizedState=t},useState:Yh,useDebugValue:Mp,useDeferredValue:function(t){return ti().memoizedState=t},useTransition:function(){var t=Yh(!1),e=t[0];return t=z_.bind(null,t[1]),ti().memoizedState=t,[e,t]},useMutableSource:function(){},useSyncExternalStore:function(t,e,n){var i=Mt,r=ti();if(yt){if(n===void 0)throw Error(fe(407));n=n()}else{if(n=e(),Ot===null)throw Error(fe(349));Hr&30||Wg(i,e,n)}r.memoizedState=n;var s={value:n,getSnapshot:e};return r.queue=s,Zh(qg.bind(null,i,s,t),[t]),i.flags|=2048,Jo(9,Xg.bind(null,i,s,n,e),void 0,null),n},useId:function(){var t=ti(),e=Ot.identifierPrefix;if(yt){var n=wi,i=Ti;n=(i&~(1<<32-qn(i)-1)).toString(32)+n,e=":"+e+"R"+n,n=Yo++,0<n&&(e+="H"+n.toString(32)),e+=":"}else n=O_++,e=":"+e+"r"+n.toString(32)+":";return t.memoizedState=e},unstable_isNewReconciler:!1},H_={readContext:Un,useCallback:t1,useContext:Un,useEffect:Sp,useImperativeHandle:e1,useInsertionEffect:Zg,useLayoutEffect:Jg,useMemo:n1,useReducer:su,useRef:Yg,useState:function(){return su(Zo)},useDebugValue:Mp,useDeferredValue:function(t){var e=kn();return i1(e,Pt.memoizedState,t)},useTransition:function(){var t=su(Zo)[0],e=kn().memoizedState;return[t,e]},useMutableSource:Hg,useSyncExternalStore:Gg,useId:r1,unstable_isNewReconciler:!1},G_={readContext:Un,useCallback:t1,useContext:Un,useEffect:Sp,useImperativeHandle:e1,useInsertionEffect:Zg,useLayoutEffect:Jg,useMemo:n1,useReducer:ou,useRef:Yg,useState:function(){return ou(Zo)},useDebugValue:Mp,useDeferredValue:function(t){var e=kn();return Pt===null?e.memoizedState=t:i1(e,Pt.memoizedState,t)},useTransition:function(){var t=ou(Zo)[0],e=kn().memoizedState;return[t,e]},useMutableSource:Hg,useSyncExternalStore:Gg,useId:r1,unstable_isNewReconciler:!1};function Hn(t,e){if(t&&t.defaultProps){e=bt({},e),t=t.defaultProps;for(var n in t)e[n]===void 0&&(e[n]=t[n]);return e}return e}function Ed(t,e,n,i){e=t.memoizedState,n=n(i,e),n=n==null?e:bt({},e,n),t.memoizedState=n,t.lanes===0&&(t.updateQueue.baseState=n)}var Sc={isMounted:function(t){return(t=t._reactInternals)?$r(t)===t:!1},enqueueSetState:function(t,e,n){t=t._reactInternals;var i=cn(),r=ar(t),s=Ai(i,r);s.payload=e,n!=null&&(s.callback=n),e=sr(t,s,r),e!==null&&($n(e,t,r,i),_l(e,t,r))},enqueueReplaceState:function(t,e,n){t=t._reactInternals;var i=cn(),r=ar(t),s=Ai(i,r);s.tag=1,s.payload=e,n!=null&&(s.callback=n),e=sr(t,s,r),e!==null&&($n(e,t,r,i),_l(e,t,r))},enqueueForceUpdate:function(t,e){t=t._reactInternals;var n=cn(),i=ar(t),r=Ai(n,i);r.tag=2,e!=null&&(r.callback=e),e=sr(t,r,i),e!==null&&($n(e,t,i,n),_l(e,t,i))}};function Jh(t,e,n,i,r,s,o){return t=t.stateNode,typeof t.shouldComponentUpdate=="function"?t.shouldComponentUpdate(i,s,o):e.prototype&&e.prototype.isPureReactComponent?!Go(n,i)||!Go(r,s):!0}function l1(t,e,n){var i=!1,r=fr,s=e.contextType;return typeof s=="object"&&s!==null?s=Un(s):(r=gn(e)?jr:nn.current,i=e.contextTypes,s=(i=i!=null)?zs(t,r):fr),e=new e(n,s),t.memoizedState=e.state!==null&&e.state!==void 0?e.state:null,e.updater=Sc,t.stateNode=e,e._reactInternals=t,i&&(t=t.stateNode,t.__reactInternalMemoizedUnmaskedChildContext=r,t.__reactInternalMemoizedMaskedChildContext=s),e}function Qh(t,e,n,i){t=e.state,typeof e.componentWillReceiveProps=="function"&&e.componentWillReceiveProps(n,i),typeof e.UNSAFE_componentWillReceiveProps=="function"&&e.UNSAFE_componentWillReceiveProps(n,i),e.state!==t&&Sc.enqueueReplaceState(e,e.state,null)}function Td(t,e,n,i){var r=t.stateNode;r.props=n,r.state=t.memoizedState,r.refs={},hp(t);var s=e.contextType;typeof s=="object"&&s!==null?r.context=Un(s):(s=gn(e)?jr:nn.current,r.context=zs(t,s)),r.state=t.memoizedState,s=e.getDerivedStateFromProps,typeof s=="function"&&(Ed(t,e,s,n),r.state=t.memoizedState),typeof e.getDerivedStateFromProps=="function"||typeof r.getSnapshotBeforeUpdate=="function"||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(e=r.state,typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount(),e!==r.state&&Sc.enqueueReplaceState(r,r.state,null),Kl(t,n,r,i),r.state=t.memoizedState),typeof r.componentDidMount=="function"&&(t.flags|=4194308)}function Hs(t,e){try{var n="",i=e;do n+=vv(i),i=i.return;while(i);var r=n}catch(s){r=`
Error generating stack: `+s.message+`
`+s.stack}return{value:t,source:e,stack:r,digest:null}}function au(t,e,n){return{value:t,source:null,stack:n??null,digest:e??null}}function wd(t,e){try{console.error(e.value)}catch(n){setTimeout(function(){throw n})}}var W_=typeof WeakMap=="function"?WeakMap:Map;function c1(t,e,n){n=Ai(-1,n),n.tag=3,n.payload={element:null};var i=e.value;return n.callback=function(){ec||(ec=!0,Ud=i),wd(t,e)},n}function u1(t,e,n){n=Ai(-1,n),n.tag=3;var i=t.type.getDerivedStateFromError;if(typeof i=="function"){var r=e.value;n.payload=function(){return i(r)},n.callback=function(){wd(t,e)}}var s=t.stateNode;return s!==null&&typeof s.componentDidCatch=="function"&&(n.callback=function(){wd(t,e),typeof i!="function"&&(or===null?or=new Set([this]):or.add(this));var o=e.stack;this.componentDidCatch(e.value,{componentStack:o!==null?o:""})}),n}function em(t,e,n){var i=t.pingCache;if(i===null){i=t.pingCache=new W_;var r=new Set;i.set(e,r)}else r=i.get(e),r===void 0&&(r=new Set,i.set(e,r));r.has(n)||(r.add(n),t=sy.bind(null,t,e,n),e.then(t,t))}function tm(t){do{var e;if((e=t.tag===13)&&(e=t.memoizedState,e=e!==null?e.dehydrated!==null:!0),e)return t;t=t.return}while(t!==null);return null}function nm(t,e,n,i,r){return t.mode&1?(t.flags|=65536,t.lanes=r,t):(t===e?t.flags|=65536:(t.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(e=Ai(-1,1),e.tag=2,sr(n,e,1))),n.lanes|=1),t)}var X_=ki.ReactCurrentOwner,hn=!1;function an(t,e,n,i){e.child=t===null?zg(e,null,n,i):js(e,t.child,n,i)}function im(t,e,n,i,r){n=n.render;var s=e.ref;return Fs(e,r),i=_p(t,e,n,i,s,r),n=yp(),t!==null&&!hn?(e.updateQueue=t.updateQueue,e.flags&=-2053,t.lanes&=~r,Fi(t,e,r)):(yt&&n&&ap(e),e.flags|=1,an(t,e,i,r),e.child)}function rm(t,e,n,i,r){if(t===null){var s=n.type;return typeof s=="function"&&!Ip(s)&&s.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(e.tag=15,e.type=s,d1(t,e,s,i,r)):(t=Tl(n.type,null,i,e,e.mode,r),t.ref=e.ref,t.return=e,e.child=t)}if(s=t.child,!(t.lanes&r)){var o=s.memoizedProps;if(n=n.compare,n=n!==null?n:Go,n(o,i)&&t.ref===e.ref)return Fi(t,e,r)}return e.flags|=1,t=lr(s,i),t.ref=e.ref,t.return=e,e.child=t}function d1(t,e,n,i,r){if(t!==null){var s=t.memoizedProps;if(Go(s,i)&&t.ref===e.ref)if(hn=!1,e.pendingProps=i=s,(t.lanes&r)!==0)t.flags&131072&&(hn=!0);else return e.lanes=t.lanes,Fi(t,e,r)}return Cd(t,e,n,i,r)}function f1(t,e,n){var i=e.pendingProps,r=i.children,s=t!==null?t.memoizedState:null;if(i.mode==="hidden")if(!(e.mode&1))e.memoizedState={baseLanes:0,cachePool:null,transitions:null},gt(Cs,Sn),Sn|=n;else{if(!(n&1073741824))return t=s!==null?s.baseLanes|n:n,e.lanes=e.childLanes=1073741824,e.memoizedState={baseLanes:t,cachePool:null,transitions:null},e.updateQueue=null,gt(Cs,Sn),Sn|=t,null;e.memoizedState={baseLanes:0,cachePool:null,transitions:null},i=s!==null?s.baseLanes:n,gt(Cs,Sn),Sn|=i}else s!==null?(i=s.baseLanes|n,e.memoizedState=null):i=n,gt(Cs,Sn),Sn|=i;return an(t,e,r,n),e.child}function p1(t,e){var n=e.ref;(t===null&&n!==null||t!==null&&t.ref!==n)&&(e.flags|=512,e.flags|=2097152)}function Cd(t,e,n,i,r){var s=gn(n)?jr:nn.current;return s=zs(e,s),Fs(e,r),n=_p(t,e,n,i,s,r),i=yp(),t!==null&&!hn?(e.updateQueue=t.updateQueue,e.flags&=-2053,t.lanes&=~r,Fi(t,e,r)):(yt&&i&&ap(e),e.flags|=1,an(t,e,n,r),e.child)}function sm(t,e,n,i,r){if(gn(n)){var s=!0;Gl(e)}else s=!1;if(Fs(e,r),e.stateNode===null)Ml(t,e),l1(e,n,i),Td(e,n,i,r),i=!0;else if(t===null){var o=e.stateNode,a=e.memoizedProps;o.props=a;var c=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=Un(u):(u=gn(n)?jr:nn.current,u=zs(e,u));var p=n.getDerivedStateFromProps,h=typeof p=="function"||typeof o.getSnapshotBeforeUpdate=="function";h||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(a!==i||c!==u)&&Qh(e,o,i,u),$i=!1;var f=e.memoizedState;o.state=f,Kl(e,i,o,r),c=e.memoizedState,a!==i||f!==c||mn.current||$i?(typeof p=="function"&&(Ed(e,n,p,i),c=e.memoizedState),(a=$i||Jh(e,n,a,i,f,c,u))?(h||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(e.flags|=4194308)):(typeof o.componentDidMount=="function"&&(e.flags|=4194308),e.memoizedProps=i,e.memoizedState=c),o.props=i,o.state=c,o.context=u,i=a):(typeof o.componentDidMount=="function"&&(e.flags|=4194308),i=!1)}else{o=e.stateNode,jg(t,e),a=e.memoizedProps,u=e.type===e.elementType?a:Hn(e.type,a),o.props=u,h=e.pendingProps,f=o.context,c=n.contextType,typeof c=="object"&&c!==null?c=Un(c):(c=gn(n)?jr:nn.current,c=zs(e,c));var g=n.getDerivedStateFromProps;(p=typeof g=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(a!==h||f!==c)&&Qh(e,o,i,c),$i=!1,f=e.memoizedState,o.state=f,Kl(e,i,o,r);var x=e.memoizedState;a!==h||f!==x||mn.current||$i?(typeof g=="function"&&(Ed(e,n,g,i),x=e.memoizedState),(u=$i||Jh(e,n,u,i,f,x,c)||!1)?(p||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(i,x,c),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(i,x,c)),typeof o.componentDidUpdate=="function"&&(e.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(e.flags|=1024)):(typeof o.componentDidUpdate!="function"||a===t.memoizedProps&&f===t.memoizedState||(e.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&f===t.memoizedState||(e.flags|=1024),e.memoizedProps=i,e.memoizedState=x),o.props=i,o.state=x,o.context=c,i=u):(typeof o.componentDidUpdate!="function"||a===t.memoizedProps&&f===t.memoizedState||(e.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&f===t.memoizedState||(e.flags|=1024),i=!1)}return Ad(t,e,n,i,s,r)}function Ad(t,e,n,i,r,s){p1(t,e);var o=(e.flags&128)!==0;if(!i&&!o)return r&&Gh(e,n,!1),Fi(t,e,s);i=e.stateNode,X_.current=e;var a=o&&typeof n.getDerivedStateFromError!="function"?null:i.render();return e.flags|=1,t!==null&&o?(e.child=js(e,t.child,null,s),e.child=js(e,null,a,s)):an(t,e,a,s),e.memoizedState=i.state,r&&Gh(e,n,!0),e.child}function h1(t){var e=t.stateNode;e.pendingContext?Hh(t,e.pendingContext,e.pendingContext!==e.context):e.context&&Hh(t,e.context,!1),mp(t,e.containerInfo)}function om(t,e,n,i,r){return Bs(),cp(r),e.flags|=256,an(t,e,n,i),e.child}var Rd={dehydrated:null,treeContext:null,retryLane:0};function Id(t){return{baseLanes:t,cachePool:null,transitions:null}}function m1(t,e,n){var i=e.pendingProps,r=St.current,s=!1,o=(e.flags&128)!==0,a;if((a=o)||(a=t!==null&&t.memoizedState===null?!1:(r&2)!==0),a?(s=!0,e.flags&=-129):(t===null||t.memoizedState!==null)&&(r|=1),gt(St,r&1),t===null)return Md(e),t=e.memoizedState,t!==null&&(t=t.dehydrated,t!==null)?(e.mode&1?t.data==="$!"?e.lanes=8:e.lanes=1073741824:e.lanes=1,null):(o=i.children,t=i.fallback,s?(i=e.mode,s=e.child,o={mode:"hidden",children:o},!(i&1)&&s!==null?(s.childLanes=0,s.pendingProps=o):s=Ec(o,i,0,null),t=zr(t,i,n,null),s.return=e,t.return=e,s.sibling=t,e.child=s,e.child.memoizedState=Id(n),e.memoizedState=Rd,t):bp(e,o));if(r=t.memoizedState,r!==null&&(a=r.dehydrated,a!==null))return q_(t,e,o,i,a,r,n);if(s){s=i.fallback,o=e.mode,r=t.child,a=r.sibling;var c={mode:"hidden",children:i.children};return!(o&1)&&e.child!==r?(i=e.child,i.childLanes=0,i.pendingProps=c,e.deletions=null):(i=lr(r,c),i.subtreeFlags=r.subtreeFlags&14680064),a!==null?s=lr(a,s):(s=zr(s,o,n,null),s.flags|=2),s.return=e,i.return=e,i.sibling=s,e.child=i,i=s,s=e.child,o=t.child.memoizedState,o=o===null?Id(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},s.memoizedState=o,s.childLanes=t.childLanes&~n,e.memoizedState=Rd,i}return s=t.child,t=s.sibling,i=lr(s,{mode:"visible",children:i.children}),!(e.mode&1)&&(i.lanes=n),i.return=e,i.sibling=null,t!==null&&(n=e.deletions,n===null?(e.deletions=[t],e.flags|=16):n.push(t)),e.child=i,e.memoizedState=null,i}function bp(t,e){return e=Ec({mode:"visible",children:e},t.mode,0,null),e.return=t,t.child=e}function La(t,e,n,i){return i!==null&&cp(i),js(e,t.child,null,n),t=bp(e,e.pendingProps.children),t.flags|=2,e.memoizedState=null,t}function q_(t,e,n,i,r,s,o){if(n)return e.flags&256?(e.flags&=-257,i=au(Error(fe(422))),La(t,e,o,i)):e.memoizedState!==null?(e.child=t.child,e.flags|=128,null):(s=i.fallback,r=e.mode,i=Ec({mode:"visible",children:i.children},r,0,null),s=zr(s,r,o,null),s.flags|=2,i.return=e,s.return=e,i.sibling=s,e.child=i,e.mode&1&&js(e,t.child,null,o),e.child.memoizedState=Id(o),e.memoizedState=Rd,s);if(!(e.mode&1))return La(t,e,o,null);if(r.data==="$!"){if(i=r.nextSibling&&r.nextSibling.dataset,i)var a=i.dgst;return i=a,s=Error(fe(419)),i=au(s,i,void 0),La(t,e,o,i)}if(a=(o&t.childLanes)!==0,hn||a){if(i=Ot,i!==null){switch(o&-o){case 4:r=2;break;case 16:r=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:r=32;break;case 536870912:r=268435456;break;default:r=0}r=r&(i.suspendedLanes|o)?0:r,r!==0&&r!==s.retryLane&&(s.retryLane=r,Li(t,r),$n(i,t,r,-1))}return Rp(),i=au(Error(fe(421))),La(t,e,o,i)}return r.data==="$?"?(e.flags|=128,e.child=t.child,e=oy.bind(null,t),r._reactRetry=e,null):(t=s.treeContext,En=rr(r.nextSibling),Tn=e,yt=!0,Wn=null,t!==null&&(Pn[Dn++]=Ti,Pn[Dn++]=wi,Pn[Dn++]=Vr,Ti=t.id,wi=t.overflow,Vr=e),e=bp(e,i.children),e.flags|=4096,e)}function am(t,e,n){t.lanes|=e;var i=t.alternate;i!==null&&(i.lanes|=e),bd(t.return,e,n)}function lu(t,e,n,i,r){var s=t.memoizedState;s===null?t.memoizedState={isBackwards:e,rendering:null,renderingStartTime:0,last:i,tail:n,tailMode:r}:(s.isBackwards=e,s.rendering=null,s.renderingStartTime=0,s.last=i,s.tail=n,s.tailMode=r)}function g1(t,e,n){var i=e.pendingProps,r=i.revealOrder,s=i.tail;if(an(t,e,i.children,n),i=St.current,i&2)i=i&1|2,e.flags|=128;else{if(t!==null&&t.flags&128)e:for(t=e.child;t!==null;){if(t.tag===13)t.memoizedState!==null&&am(t,n,e);else if(t.tag===19)am(t,n,e);else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break e;for(;t.sibling===null;){if(t.return===null||t.return===e)break e;t=t.return}t.sibling.return=t.return,t=t.sibling}i&=1}if(gt(St,i),!(e.mode&1))e.memoizedState=null;else switch(r){case"forwards":for(n=e.child,r=null;n!==null;)t=n.alternate,t!==null&&Yl(t)===null&&(r=n),n=n.sibling;n=r,n===null?(r=e.child,e.child=null):(r=n.sibling,n.sibling=null),lu(e,!1,r,n,s);break;case"backwards":for(n=null,r=e.child,e.child=null;r!==null;){if(t=r.alternate,t!==null&&Yl(t)===null){e.child=r;break}t=r.sibling,r.sibling=n,n=r,r=t}lu(e,!0,n,null,s);break;case"together":lu(e,!1,null,null,void 0);break;default:e.memoizedState=null}return e.child}function Ml(t,e){!(e.mode&1)&&t!==null&&(t.alternate=null,e.alternate=null,e.flags|=2)}function Fi(t,e,n){if(t!==null&&(e.dependencies=t.dependencies),Gr|=e.lanes,!(n&e.childLanes))return null;if(t!==null&&e.child!==t.child)throw Error(fe(153));if(e.child!==null){for(t=e.child,n=lr(t,t.pendingProps),e.child=n,n.return=e;t.sibling!==null;)t=t.sibling,n=n.sibling=lr(t,t.pendingProps),n.return=e;n.sibling=null}return e.child}function $_(t,e,n){switch(e.tag){case 3:h1(e),Bs();break;case 5:Vg(e);break;case 1:gn(e.type)&&Gl(e);break;case 4:mp(e,e.stateNode.containerInfo);break;case 10:var i=e.type._context,r=e.memoizedProps.value;gt(ql,i._currentValue),i._currentValue=r;break;case 13:if(i=e.memoizedState,i!==null)return i.dehydrated!==null?(gt(St,St.current&1),e.flags|=128,null):n&e.child.childLanes?m1(t,e,n):(gt(St,St.current&1),t=Fi(t,e,n),t!==null?t.sibling:null);gt(St,St.current&1);break;case 19:if(i=(n&e.childLanes)!==0,t.flags&128){if(i)return g1(t,e,n);e.flags|=128}if(r=e.memoizedState,r!==null&&(r.rendering=null,r.tail=null,r.lastEffect=null),gt(St,St.current),i)break;return null;case 22:case 23:return e.lanes=0,f1(t,e,n)}return Fi(t,e,n)}var x1,Pd,v1,_1;x1=function(t,e){for(var n=e.child;n!==null;){if(n.tag===5||n.tag===6)t.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===e)break;for(;n.sibling===null;){if(n.return===null||n.return===e)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};Pd=function(){};v1=function(t,e,n,i){var r=t.memoizedProps;if(r!==i){t=e.stateNode,Nr(ci.current);var s=null;switch(n){case"input":r=Qu(t,r),i=Qu(t,i),s=[];break;case"select":r=bt({},r,{value:void 0}),i=bt({},i,{value:void 0}),s=[];break;case"textarea":r=nd(t,r),i=nd(t,i),s=[];break;default:typeof r.onClick!="function"&&typeof i.onClick=="function"&&(t.onclick=Vl)}rd(n,i);var o;n=null;for(u in r)if(!i.hasOwnProperty(u)&&r.hasOwnProperty(u)&&r[u]!=null)if(u==="style"){var a=r[u];for(o in a)a.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(ko.hasOwnProperty(u)?s||(s=[]):(s=s||[]).push(u,null));for(u in i){var c=i[u];if(a=r!=null?r[u]:void 0,i.hasOwnProperty(u)&&c!==a&&(c!=null||a!=null))if(u==="style")if(a){for(o in a)!a.hasOwnProperty(o)||c&&c.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in c)c.hasOwnProperty(o)&&a[o]!==c[o]&&(n||(n={}),n[o]=c[o])}else n||(s||(s=[]),s.push(u,n)),n=c;else u==="dangerouslySetInnerHTML"?(c=c?c.__html:void 0,a=a?a.__html:void 0,c!=null&&a!==c&&(s=s||[]).push(u,c)):u==="children"?typeof c!="string"&&typeof c!="number"||(s=s||[]).push(u,""+c):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(ko.hasOwnProperty(u)?(c!=null&&u==="onScroll"&&xt("scroll",t),s||a===c||(s=[])):(s=s||[]).push(u,c))}n&&(s=s||[]).push("style",n);var u=s;(e.updateQueue=u)&&(e.flags|=4)}};_1=function(t,e,n,i){n!==i&&(e.flags|=4)};function co(t,e){if(!yt)switch(t.tailMode){case"hidden":e=t.tail;for(var n=null;e!==null;)e.alternate!==null&&(n=e),e=e.sibling;n===null?t.tail=null:n.sibling=null;break;case"collapsed":n=t.tail;for(var i=null;n!==null;)n.alternate!==null&&(i=n),n=n.sibling;i===null?e||t.tail===null?t.tail=null:t.tail.sibling=null:i.sibling=null}}function Kt(t){var e=t.alternate!==null&&t.alternate.child===t.child,n=0,i=0;if(e)for(var r=t.child;r!==null;)n|=r.lanes|r.childLanes,i|=r.subtreeFlags&14680064,i|=r.flags&14680064,r.return=t,r=r.sibling;else for(r=t.child;r!==null;)n|=r.lanes|r.childLanes,i|=r.subtreeFlags,i|=r.flags,r.return=t,r=r.sibling;return t.subtreeFlags|=i,t.childLanes=n,e}function K_(t,e,n){var i=e.pendingProps;switch(lp(e),e.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Kt(e),null;case 1:return gn(e.type)&&Hl(),Kt(e),null;case 3:return i=e.stateNode,Vs(),vt(mn),vt(nn),xp(),i.pendingContext&&(i.context=i.pendingContext,i.pendingContext=null),(t===null||t.child===null)&&(Pa(e)?e.flags|=4:t===null||t.memoizedState.isDehydrated&&!(e.flags&256)||(e.flags|=1024,Wn!==null&&(zd(Wn),Wn=null))),Pd(t,e),Kt(e),null;case 5:gp(e);var r=Nr(Ko.current);if(n=e.type,t!==null&&e.stateNode!=null)v1(t,e,n,i,r),t.ref!==e.ref&&(e.flags|=512,e.flags|=2097152);else{if(!i){if(e.stateNode===null)throw Error(fe(166));return Kt(e),null}if(t=Nr(ci.current),Pa(e)){i=e.stateNode,n=e.type;var s=e.memoizedProps;switch(i[ii]=e,i[qo]=s,t=(e.mode&1)!==0,n){case"dialog":xt("cancel",i),xt("close",i);break;case"iframe":case"object":case"embed":xt("load",i);break;case"video":case"audio":for(r=0;r<To.length;r++)xt(To[r],i);break;case"source":xt("error",i);break;case"img":case"image":case"link":xt("error",i),xt("load",i);break;case"details":xt("toggle",i);break;case"input":gh(i,s),xt("invalid",i);break;case"select":i._wrapperState={wasMultiple:!!s.multiple},xt("invalid",i);break;case"textarea":vh(i,s),xt("invalid",i)}rd(n,s),r=null;for(var o in s)if(s.hasOwnProperty(o)){var a=s[o];o==="children"?typeof a=="string"?i.textContent!==a&&(s.suppressHydrationWarning!==!0&&Ia(i.textContent,a,t),r=["children",a]):typeof a=="number"&&i.textContent!==""+a&&(s.suppressHydrationWarning!==!0&&Ia(i.textContent,a,t),r=["children",""+a]):ko.hasOwnProperty(o)&&a!=null&&o==="onScroll"&&xt("scroll",i)}switch(n){case"input":Ma(i),xh(i,s,!0);break;case"textarea":Ma(i),_h(i);break;case"select":case"option":break;default:typeof s.onClick=="function"&&(i.onclick=Vl)}i=r,e.updateQueue=i,i!==null&&(e.flags|=4)}else{o=r.nodeType===9?r:r.ownerDocument,t==="http://www.w3.org/1999/xhtml"&&(t=q0(n)),t==="http://www.w3.org/1999/xhtml"?n==="script"?(t=o.createElement("div"),t.innerHTML="<script><\/script>",t=t.removeChild(t.firstChild)):typeof i.is=="string"?t=o.createElement(n,{is:i.is}):(t=o.createElement(n),n==="select"&&(o=t,i.multiple?o.multiple=!0:i.size&&(o.size=i.size))):t=o.createElementNS(t,n),t[ii]=e,t[qo]=i,x1(t,e,!1,!1),e.stateNode=t;e:{switch(o=sd(n,i),n){case"dialog":xt("cancel",t),xt("close",t),r=i;break;case"iframe":case"object":case"embed":xt("load",t),r=i;break;case"video":case"audio":for(r=0;r<To.length;r++)xt(To[r],t);r=i;break;case"source":xt("error",t),r=i;break;case"img":case"image":case"link":xt("error",t),xt("load",t),r=i;break;case"details":xt("toggle",t),r=i;break;case"input":gh(t,i),r=Qu(t,i),xt("invalid",t);break;case"option":r=i;break;case"select":t._wrapperState={wasMultiple:!!i.multiple},r=bt({},i,{value:void 0}),xt("invalid",t);break;case"textarea":vh(t,i),r=nd(t,i),xt("invalid",t);break;default:r=i}rd(n,r),a=r;for(s in a)if(a.hasOwnProperty(s)){var c=a[s];s==="style"?Y0(t,c):s==="dangerouslySetInnerHTML"?(c=c?c.__html:void 0,c!=null&&$0(t,c)):s==="children"?typeof c=="string"?(n!=="textarea"||c!=="")&&Oo(t,c):typeof c=="number"&&Oo(t,""+c):s!=="suppressContentEditableWarning"&&s!=="suppressHydrationWarning"&&s!=="autoFocus"&&(ko.hasOwnProperty(s)?c!=null&&s==="onScroll"&&xt("scroll",t):c!=null&&qf(t,s,c,o))}switch(n){case"input":Ma(t),xh(t,i,!1);break;case"textarea":Ma(t),_h(t);break;case"option":i.value!=null&&t.setAttribute("value",""+dr(i.value));break;case"select":t.multiple=!!i.multiple,s=i.value,s!=null?Is(t,!!i.multiple,s,!1):i.defaultValue!=null&&Is(t,!!i.multiple,i.defaultValue,!0);break;default:typeof r.onClick=="function"&&(t.onclick=Vl)}switch(n){case"button":case"input":case"select":case"textarea":i=!!i.autoFocus;break e;case"img":i=!0;break e;default:i=!1}}i&&(e.flags|=4)}e.ref!==null&&(e.flags|=512,e.flags|=2097152)}return Kt(e),null;case 6:if(t&&e.stateNode!=null)_1(t,e,t.memoizedProps,i);else{if(typeof i!="string"&&e.stateNode===null)throw Error(fe(166));if(n=Nr(Ko.current),Nr(ci.current),Pa(e)){if(i=e.stateNode,n=e.memoizedProps,i[ii]=e,(s=i.nodeValue!==n)&&(t=Tn,t!==null))switch(t.tag){case 3:Ia(i.nodeValue,n,(t.mode&1)!==0);break;case 5:t.memoizedProps.suppressHydrationWarning!==!0&&Ia(i.nodeValue,n,(t.mode&1)!==0)}s&&(e.flags|=4)}else i=(n.nodeType===9?n:n.ownerDocument).createTextNode(i),i[ii]=e,e.stateNode=i}return Kt(e),null;case 13:if(vt(St),i=e.memoizedState,t===null||t.memoizedState!==null&&t.memoizedState.dehydrated!==null){if(yt&&En!==null&&e.mode&1&&!(e.flags&128))kg(),Bs(),e.flags|=98560,s=!1;else if(s=Pa(e),i!==null&&i.dehydrated!==null){if(t===null){if(!s)throw Error(fe(318));if(s=e.memoizedState,s=s!==null?s.dehydrated:null,!s)throw Error(fe(317));s[ii]=e}else Bs(),!(e.flags&128)&&(e.memoizedState=null),e.flags|=4;Kt(e),s=!1}else Wn!==null&&(zd(Wn),Wn=null),s=!0;if(!s)return e.flags&65536?e:null}return e.flags&128?(e.lanes=n,e):(i=i!==null,i!==(t!==null&&t.memoizedState!==null)&&i&&(e.child.flags|=8192,e.mode&1&&(t===null||St.current&1?Dt===0&&(Dt=3):Rp())),e.updateQueue!==null&&(e.flags|=4),Kt(e),null);case 4:return Vs(),Pd(t,e),t===null&&Wo(e.stateNode.containerInfo),Kt(e),null;case 10:return fp(e.type._context),Kt(e),null;case 17:return gn(e.type)&&Hl(),Kt(e),null;case 19:if(vt(St),s=e.memoizedState,s===null)return Kt(e),null;if(i=(e.flags&128)!==0,o=s.rendering,o===null)if(i)co(s,!1);else{if(Dt!==0||t!==null&&t.flags&128)for(t=e.child;t!==null;){if(o=Yl(t),o!==null){for(e.flags|=128,co(s,!1),i=o.updateQueue,i!==null&&(e.updateQueue=i,e.flags|=4),e.subtreeFlags=0,i=n,n=e.child;n!==null;)s=n,t=i,s.flags&=14680066,o=s.alternate,o===null?(s.childLanes=0,s.lanes=t,s.child=null,s.subtreeFlags=0,s.memoizedProps=null,s.memoizedState=null,s.updateQueue=null,s.dependencies=null,s.stateNode=null):(s.childLanes=o.childLanes,s.lanes=o.lanes,s.child=o.child,s.subtreeFlags=0,s.deletions=null,s.memoizedProps=o.memoizedProps,s.memoizedState=o.memoizedState,s.updateQueue=o.updateQueue,s.type=o.type,t=o.dependencies,s.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),n=n.sibling;return gt(St,St.current&1|2),e.child}t=t.sibling}s.tail!==null&&At()>Gs&&(e.flags|=128,i=!0,co(s,!1),e.lanes=4194304)}else{if(!i)if(t=Yl(o),t!==null){if(e.flags|=128,i=!0,n=t.updateQueue,n!==null&&(e.updateQueue=n,e.flags|=4),co(s,!0),s.tail===null&&s.tailMode==="hidden"&&!o.alternate&&!yt)return Kt(e),null}else 2*At()-s.renderingStartTime>Gs&&n!==1073741824&&(e.flags|=128,i=!0,co(s,!1),e.lanes=4194304);s.isBackwards?(o.sibling=e.child,e.child=o):(n=s.last,n!==null?n.sibling=o:e.child=o,s.last=o)}return s.tail!==null?(e=s.tail,s.rendering=e,s.tail=e.sibling,s.renderingStartTime=At(),e.sibling=null,n=St.current,gt(St,i?n&1|2:n&1),e):(Kt(e),null);case 22:case 23:return Ap(),i=e.memoizedState!==null,t!==null&&t.memoizedState!==null!==i&&(e.flags|=8192),i&&e.mode&1?Sn&1073741824&&(Kt(e),e.subtreeFlags&6&&(e.flags|=8192)):Kt(e),null;case 24:return null;case 25:return null}throw Error(fe(156,e.tag))}function Y_(t,e){switch(lp(e),e.tag){case 1:return gn(e.type)&&Hl(),t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 3:return Vs(),vt(mn),vt(nn),xp(),t=e.flags,t&65536&&!(t&128)?(e.flags=t&-65537|128,e):null;case 5:return gp(e),null;case 13:if(vt(St),t=e.memoizedState,t!==null&&t.dehydrated!==null){if(e.alternate===null)throw Error(fe(340));Bs()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 19:return vt(St),null;case 4:return Vs(),null;case 10:return fp(e.type._context),null;case 22:case 23:return Ap(),null;case 24:return null;default:return null}}var Fa=!1,Jt=!1,Z_=typeof WeakSet=="function"?WeakSet:Set,Ce=null;function ws(t,e){var n=t.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(i){Tt(t,e,i)}else n.current=null}function Dd(t,e,n){try{n()}catch(i){Tt(t,e,i)}}var lm=!1;function J_(t,e){if(md=zl,t=Eg(),op(t)){if("selectionStart"in t)var n={start:t.selectionStart,end:t.selectionEnd};else e:{n=(n=t.ownerDocument)&&n.defaultView||window;var i=n.getSelection&&n.getSelection();if(i&&i.rangeCount!==0){n=i.anchorNode;var r=i.anchorOffset,s=i.focusNode;i=i.focusOffset;try{n.nodeType,s.nodeType}catch{n=null;break e}var o=0,a=-1,c=-1,u=0,p=0,h=t,f=null;t:for(;;){for(var g;h!==n||r!==0&&h.nodeType!==3||(a=o+r),h!==s||i!==0&&h.nodeType!==3||(c=o+i),h.nodeType===3&&(o+=h.nodeValue.length),(g=h.firstChild)!==null;)f=h,h=g;for(;;){if(h===t)break t;if(f===n&&++u===r&&(a=o),f===s&&++p===i&&(c=o),(g=h.nextSibling)!==null)break;h=f,f=h.parentNode}h=g}n=a===-1||c===-1?null:{start:a,end:c}}else n=null}n=n||{start:0,end:0}}else n=null;for(gd={focusedElem:t,selectionRange:n},zl=!1,Ce=e;Ce!==null;)if(e=Ce,t=e.child,(e.subtreeFlags&1028)!==0&&t!==null)t.return=e,Ce=t;else for(;Ce!==null;){e=Ce;try{var x=e.alternate;if(e.flags&1024)switch(e.tag){case 0:case 11:case 15:break;case 1:if(x!==null){var M=x.memoizedProps,v=x.memoizedState,d=e.stateNode,m=d.getSnapshotBeforeUpdate(e.elementType===e.type?M:Hn(e.type,M),v);d.__reactInternalSnapshotBeforeUpdate=m}break;case 3:var _=e.stateNode.containerInfo;_.nodeType===1?_.textContent="":_.nodeType===9&&_.documentElement&&_.removeChild(_.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(fe(163))}}catch(b){Tt(e,e.return,b)}if(t=e.sibling,t!==null){t.return=e.return,Ce=t;break}Ce=e.return}return x=lm,lm=!1,x}function Fo(t,e,n){var i=e.updateQueue;if(i=i!==null?i.lastEffect:null,i!==null){var r=i=i.next;do{if((r.tag&t)===t){var s=r.destroy;r.destroy=void 0,s!==void 0&&Dd(e,n,s)}r=r.next}while(r!==i)}}function Mc(t,e){if(e=e.updateQueue,e=e!==null?e.lastEffect:null,e!==null){var n=e=e.next;do{if((n.tag&t)===t){var i=n.create;n.destroy=i()}n=n.next}while(n!==e)}}function Ld(t){var e=t.ref;if(e!==null){var n=t.stateNode;switch(t.tag){case 5:t=n;break;default:t=n}typeof e=="function"?e(t):e.current=t}}function y1(t){var e=t.alternate;e!==null&&(t.alternate=null,y1(e)),t.child=null,t.deletions=null,t.sibling=null,t.tag===5&&(e=t.stateNode,e!==null&&(delete e[ii],delete e[qo],delete e[_d],delete e[F_],delete e[N_])),t.stateNode=null,t.return=null,t.dependencies=null,t.memoizedProps=null,t.memoizedState=null,t.pendingProps=null,t.stateNode=null,t.updateQueue=null}function S1(t){return t.tag===5||t.tag===3||t.tag===4}function cm(t){e:for(;;){for(;t.sibling===null;){if(t.return===null||S1(t.return))return null;t=t.return}for(t.sibling.return=t.return,t=t.sibling;t.tag!==5&&t.tag!==6&&t.tag!==18;){if(t.flags&2||t.child===null||t.tag===4)continue e;t.child.return=t,t=t.child}if(!(t.flags&2))return t.stateNode}}function Fd(t,e,n){var i=t.tag;if(i===5||i===6)t=t.stateNode,e?n.nodeType===8?n.parentNode.insertBefore(t,e):n.insertBefore(t,e):(n.nodeType===8?(e=n.parentNode,e.insertBefore(t,n)):(e=n,e.appendChild(t)),n=n._reactRootContainer,n!=null||e.onclick!==null||(e.onclick=Vl));else if(i!==4&&(t=t.child,t!==null))for(Fd(t,e,n),t=t.sibling;t!==null;)Fd(t,e,n),t=t.sibling}function Nd(t,e,n){var i=t.tag;if(i===5||i===6)t=t.stateNode,e?n.insertBefore(t,e):n.appendChild(t);else if(i!==4&&(t=t.child,t!==null))for(Nd(t,e,n),t=t.sibling;t!==null;)Nd(t,e,n),t=t.sibling}var Bt=null,Gn=!1;function Bi(t,e,n){for(n=n.child;n!==null;)M1(t,e,n),n=n.sibling}function M1(t,e,n){if(li&&typeof li.onCommitFiberUnmount=="function")try{li.onCommitFiberUnmount(hc,n)}catch{}switch(n.tag){case 5:Jt||ws(n,e);case 6:var i=Bt,r=Gn;Bt=null,Bi(t,e,n),Bt=i,Gn=r,Bt!==null&&(Gn?(t=Bt,n=n.stateNode,t.nodeType===8?t.parentNode.removeChild(n):t.removeChild(n)):Bt.removeChild(n.stateNode));break;case 18:Bt!==null&&(Gn?(t=Bt,n=n.stateNode,t.nodeType===8?tu(t.parentNode,n):t.nodeType===1&&tu(t,n),Vo(t)):tu(Bt,n.stateNode));break;case 4:i=Bt,r=Gn,Bt=n.stateNode.containerInfo,Gn=!0,Bi(t,e,n),Bt=i,Gn=r;break;case 0:case 11:case 14:case 15:if(!Jt&&(i=n.updateQueue,i!==null&&(i=i.lastEffect,i!==null))){r=i=i.next;do{var s=r,o=s.destroy;s=s.tag,o!==void 0&&(s&2||s&4)&&Dd(n,e,o),r=r.next}while(r!==i)}Bi(t,e,n);break;case 1:if(!Jt&&(ws(n,e),i=n.stateNode,typeof i.componentWillUnmount=="function"))try{i.props=n.memoizedProps,i.state=n.memoizedState,i.componentWillUnmount()}catch(a){Tt(n,e,a)}Bi(t,e,n);break;case 21:Bi(t,e,n);break;case 22:n.mode&1?(Jt=(i=Jt)||n.memoizedState!==null,Bi(t,e,n),Jt=i):Bi(t,e,n);break;default:Bi(t,e,n)}}function um(t){var e=t.updateQueue;if(e!==null){t.updateQueue=null;var n=t.stateNode;n===null&&(n=t.stateNode=new Z_),e.forEach(function(i){var r=ay.bind(null,t,i);n.has(i)||(n.add(i),i.then(r,r))})}}function zn(t,e){var n=e.deletions;if(n!==null)for(var i=0;i<n.length;i++){var r=n[i];try{var s=t,o=e,a=o;e:for(;a!==null;){switch(a.tag){case 5:Bt=a.stateNode,Gn=!1;break e;case 3:Bt=a.stateNode.containerInfo,Gn=!0;break e;case 4:Bt=a.stateNode.containerInfo,Gn=!0;break e}a=a.return}if(Bt===null)throw Error(fe(160));M1(s,o,r),Bt=null,Gn=!1;var c=r.alternate;c!==null&&(c.return=null),r.return=null}catch(u){Tt(r,e,u)}}if(e.subtreeFlags&12854)for(e=e.child;e!==null;)b1(e,t),e=e.sibling}function b1(t,e){var n=t.alternate,i=t.flags;switch(t.tag){case 0:case 11:case 14:case 15:if(zn(e,t),Jn(t),i&4){try{Fo(3,t,t.return),Mc(3,t)}catch(M){Tt(t,t.return,M)}try{Fo(5,t,t.return)}catch(M){Tt(t,t.return,M)}}break;case 1:zn(e,t),Jn(t),i&512&&n!==null&&ws(n,n.return);break;case 5:if(zn(e,t),Jn(t),i&512&&n!==null&&ws(n,n.return),t.flags&32){var r=t.stateNode;try{Oo(r,"")}catch(M){Tt(t,t.return,M)}}if(i&4&&(r=t.stateNode,r!=null)){var s=t.memoizedProps,o=n!==null?n.memoizedProps:s,a=t.type,c=t.updateQueue;if(t.updateQueue=null,c!==null)try{a==="input"&&s.type==="radio"&&s.name!=null&&W0(r,s),sd(a,o);var u=sd(a,s);for(o=0;o<c.length;o+=2){var p=c[o],h=c[o+1];p==="style"?Y0(r,h):p==="dangerouslySetInnerHTML"?$0(r,h):p==="children"?Oo(r,h):qf(r,p,h,u)}switch(a){case"input":ed(r,s);break;case"textarea":X0(r,s);break;case"select":var f=r._wrapperState.wasMultiple;r._wrapperState.wasMultiple=!!s.multiple;var g=s.value;g!=null?Is(r,!!s.multiple,g,!1):f!==!!s.multiple&&(s.defaultValue!=null?Is(r,!!s.multiple,s.defaultValue,!0):Is(r,!!s.multiple,s.multiple?[]:"",!1))}r[qo]=s}catch(M){Tt(t,t.return,M)}}break;case 6:if(zn(e,t),Jn(t),i&4){if(t.stateNode===null)throw Error(fe(162));r=t.stateNode,s=t.memoizedProps;try{r.nodeValue=s}catch(M){Tt(t,t.return,M)}}break;case 3:if(zn(e,t),Jn(t),i&4&&n!==null&&n.memoizedState.isDehydrated)try{Vo(e.containerInfo)}catch(M){Tt(t,t.return,M)}break;case 4:zn(e,t),Jn(t);break;case 13:zn(e,t),Jn(t),r=t.child,r.flags&8192&&(s=r.memoizedState!==null,r.stateNode.isHidden=s,!s||r.alternate!==null&&r.alternate.memoizedState!==null||(wp=At())),i&4&&um(t);break;case 22:if(p=n!==null&&n.memoizedState!==null,t.mode&1?(Jt=(u=Jt)||p,zn(e,t),Jt=u):zn(e,t),Jn(t),i&8192){if(u=t.memoizedState!==null,(t.stateNode.isHidden=u)&&!p&&t.mode&1)for(Ce=t,p=t.child;p!==null;){for(h=Ce=p;Ce!==null;){switch(f=Ce,g=f.child,f.tag){case 0:case 11:case 14:case 15:Fo(4,f,f.return);break;case 1:ws(f,f.return);var x=f.stateNode;if(typeof x.componentWillUnmount=="function"){i=f,n=f.return;try{e=i,x.props=e.memoizedProps,x.state=e.memoizedState,x.componentWillUnmount()}catch(M){Tt(i,n,M)}}break;case 5:ws(f,f.return);break;case 22:if(f.memoizedState!==null){fm(h);continue}}g!==null?(g.return=f,Ce=g):fm(h)}p=p.sibling}e:for(p=null,h=t;;){if(h.tag===5){if(p===null){p=h;try{r=h.stateNode,u?(s=r.style,typeof s.setProperty=="function"?s.setProperty("display","none","important"):s.display="none"):(a=h.stateNode,c=h.memoizedProps.style,o=c!=null&&c.hasOwnProperty("display")?c.display:null,a.style.display=K0("display",o))}catch(M){Tt(t,t.return,M)}}}else if(h.tag===6){if(p===null)try{h.stateNode.nodeValue=u?"":h.memoizedProps}catch(M){Tt(t,t.return,M)}}else if((h.tag!==22&&h.tag!==23||h.memoizedState===null||h===t)&&h.child!==null){h.child.return=h,h=h.child;continue}if(h===t)break e;for(;h.sibling===null;){if(h.return===null||h.return===t)break e;p===h&&(p=null),h=h.return}p===h&&(p=null),h.sibling.return=h.return,h=h.sibling}}break;case 19:zn(e,t),Jn(t),i&4&&um(t);break;case 21:break;default:zn(e,t),Jn(t)}}function Jn(t){var e=t.flags;if(e&2){try{e:{for(var n=t.return;n!==null;){if(S1(n)){var i=n;break e}n=n.return}throw Error(fe(160))}switch(i.tag){case 5:var r=i.stateNode;i.flags&32&&(Oo(r,""),i.flags&=-33);var s=cm(t);Nd(t,s,r);break;case 3:case 4:var o=i.stateNode.containerInfo,a=cm(t);Fd(t,a,o);break;default:throw Error(fe(161))}}catch(c){Tt(t,t.return,c)}t.flags&=-3}e&4096&&(t.flags&=-4097)}function Q_(t,e,n){Ce=t,E1(t)}function E1(t,e,n){for(var i=(t.mode&1)!==0;Ce!==null;){var r=Ce,s=r.child;if(r.tag===22&&i){var o=r.memoizedState!==null||Fa;if(!o){var a=r.alternate,c=a!==null&&a.memoizedState!==null||Jt;a=Fa;var u=Jt;if(Fa=o,(Jt=c)&&!u)for(Ce=r;Ce!==null;)o=Ce,c=o.child,o.tag===22&&o.memoizedState!==null?pm(r):c!==null?(c.return=o,Ce=c):pm(r);for(;s!==null;)Ce=s,E1(s),s=s.sibling;Ce=r,Fa=a,Jt=u}dm(t)}else r.subtreeFlags&8772&&s!==null?(s.return=r,Ce=s):dm(t)}}function dm(t){for(;Ce!==null;){var e=Ce;if(e.flags&8772){var n=e.alternate;try{if(e.flags&8772)switch(e.tag){case 0:case 11:case 15:Jt||Mc(5,e);break;case 1:var i=e.stateNode;if(e.flags&4&&!Jt)if(n===null)i.componentDidMount();else{var r=e.elementType===e.type?n.memoizedProps:Hn(e.type,n.memoizedProps);i.componentDidUpdate(r,n.memoizedState,i.__reactInternalSnapshotBeforeUpdate)}var s=e.updateQueue;s!==null&&Kh(e,s,i);break;case 3:var o=e.updateQueue;if(o!==null){if(n=null,e.child!==null)switch(e.child.tag){case 5:n=e.child.stateNode;break;case 1:n=e.child.stateNode}Kh(e,o,n)}break;case 5:var a=e.stateNode;if(n===null&&e.flags&4){n=a;var c=e.memoizedProps;switch(e.type){case"button":case"input":case"select":case"textarea":c.autoFocus&&n.focus();break;case"img":c.src&&(n.src=c.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(e.memoizedState===null){var u=e.alternate;if(u!==null){var p=u.memoizedState;if(p!==null){var h=p.dehydrated;h!==null&&Vo(h)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(fe(163))}Jt||e.flags&512&&Ld(e)}catch(f){Tt(e,e.return,f)}}if(e===t){Ce=null;break}if(n=e.sibling,n!==null){n.return=e.return,Ce=n;break}Ce=e.return}}function fm(t){for(;Ce!==null;){var e=Ce;if(e===t){Ce=null;break}var n=e.sibling;if(n!==null){n.return=e.return,Ce=n;break}Ce=e.return}}function pm(t){for(;Ce!==null;){var e=Ce;try{switch(e.tag){case 0:case 11:case 15:var n=e.return;try{Mc(4,e)}catch(c){Tt(e,n,c)}break;case 1:var i=e.stateNode;if(typeof i.componentDidMount=="function"){var r=e.return;try{i.componentDidMount()}catch(c){Tt(e,r,c)}}var s=e.return;try{Ld(e)}catch(c){Tt(e,s,c)}break;case 5:var o=e.return;try{Ld(e)}catch(c){Tt(e,o,c)}}}catch(c){Tt(e,e.return,c)}if(e===t){Ce=null;break}var a=e.sibling;if(a!==null){a.return=e.return,Ce=a;break}Ce=e.return}}var ey=Math.ceil,Ql=ki.ReactCurrentDispatcher,Ep=ki.ReactCurrentOwner,Nn=ki.ReactCurrentBatchConfig,st=0,Ot=null,It=null,Ht=0,Sn=0,Cs=gr(0),Dt=0,Qo=null,Gr=0,bc=0,Tp=0,No=null,fn=null,wp=0,Gs=1/0,bi=null,ec=!1,Ud=null,or=null,Na=!1,Qi=null,tc=0,Uo=0,kd=null,bl=-1,El=0;function cn(){return st&6?At():bl!==-1?bl:bl=At()}function ar(t){return t.mode&1?st&2&&Ht!==0?Ht&-Ht:k_.transition!==null?(El===0&&(El=lg()),El):(t=ct,t!==0||(t=window.event,t=t===void 0?16:mg(t.type)),t):1}function $n(t,e,n,i){if(50<Uo)throw Uo=0,kd=null,Error(fe(185));aa(t,n,i),(!(st&2)||t!==Ot)&&(t===Ot&&(!(st&2)&&(bc|=n),Dt===4&&Yi(t,Ht)),xn(t,i),n===1&&st===0&&!(e.mode&1)&&(Gs=At()+500,_c&&xr()))}function xn(t,e){var n=t.callbackNode;kv(t,e);var i=Ol(t,t===Ot?Ht:0);if(i===0)n!==null&&Mh(n),t.callbackNode=null,t.callbackPriority=0;else if(e=i&-i,t.callbackPriority!==e){if(n!=null&&Mh(n),e===1)t.tag===0?U_(hm.bind(null,t)):Fg(hm.bind(null,t)),D_(function(){!(st&6)&&xr()}),n=null;else{switch(cg(i)){case 1:n=Jf;break;case 4:n=og;break;case 16:n=kl;break;case 536870912:n=ag;break;default:n=kl}n=D1(n,T1.bind(null,t))}t.callbackPriority=e,t.callbackNode=n}}function T1(t,e){if(bl=-1,El=0,st&6)throw Error(fe(327));var n=t.callbackNode;if(Ns()&&t.callbackNode!==n)return null;var i=Ol(t,t===Ot?Ht:0);if(i===0)return null;if(i&30||i&t.expiredLanes||e)e=nc(t,i);else{e=i;var r=st;st|=2;var s=C1();(Ot!==t||Ht!==e)&&(bi=null,Gs=At()+500,Or(t,e));do try{iy();break}catch(a){w1(t,a)}while(!0);dp(),Ql.current=s,st=r,It!==null?e=0:(Ot=null,Ht=0,e=Dt)}if(e!==0){if(e===2&&(r=ud(t),r!==0&&(i=r,e=Od(t,r))),e===1)throw n=Qo,Or(t,0),Yi(t,i),xn(t,At()),n;if(e===6)Yi(t,i);else{if(r=t.current.alternate,!(i&30)&&!ty(r)&&(e=nc(t,i),e===2&&(s=ud(t),s!==0&&(i=s,e=Od(t,s))),e===1))throw n=Qo,Or(t,0),Yi(t,i),xn(t,At()),n;switch(t.finishedWork=r,t.finishedLanes=i,e){case 0:case 1:throw Error(fe(345));case 2:Rr(t,fn,bi);break;case 3:if(Yi(t,i),(i&130023424)===i&&(e=wp+500-At(),10<e)){if(Ol(t,0)!==0)break;if(r=t.suspendedLanes,(r&i)!==i){cn(),t.pingedLanes|=t.suspendedLanes&r;break}t.timeoutHandle=vd(Rr.bind(null,t,fn,bi),e);break}Rr(t,fn,bi);break;case 4:if(Yi(t,i),(i&4194240)===i)break;for(e=t.eventTimes,r=-1;0<i;){var o=31-qn(i);s=1<<o,o=e[o],o>r&&(r=o),i&=~s}if(i=r,i=At()-i,i=(120>i?120:480>i?480:1080>i?1080:1920>i?1920:3e3>i?3e3:4320>i?4320:1960*ey(i/1960))-i,10<i){t.timeoutHandle=vd(Rr.bind(null,t,fn,bi),i);break}Rr(t,fn,bi);break;case 5:Rr(t,fn,bi);break;default:throw Error(fe(329))}}}return xn(t,At()),t.callbackNode===n?T1.bind(null,t):null}function Od(t,e){var n=No;return t.current.memoizedState.isDehydrated&&(Or(t,e).flags|=256),t=nc(t,e),t!==2&&(e=fn,fn=n,e!==null&&zd(e)),t}function zd(t){fn===null?fn=t:fn.push.apply(fn,t)}function ty(t){for(var e=t;;){if(e.flags&16384){var n=e.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var i=0;i<n.length;i++){var r=n[i],s=r.getSnapshot;r=r.value;try{if(!Yn(s(),r))return!1}catch{return!1}}}if(n=e.child,e.subtreeFlags&16384&&n!==null)n.return=e,e=n;else{if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return!0;e=e.return}e.sibling.return=e.return,e=e.sibling}}return!0}function Yi(t,e){for(e&=~Tp,e&=~bc,t.suspendedLanes|=e,t.pingedLanes&=~e,t=t.expirationTimes;0<e;){var n=31-qn(e),i=1<<n;t[n]=-1,e&=~i}}function hm(t){if(st&6)throw Error(fe(327));Ns();var e=Ol(t,0);if(!(e&1))return xn(t,At()),null;var n=nc(t,e);if(t.tag!==0&&n===2){var i=ud(t);i!==0&&(e=i,n=Od(t,i))}if(n===1)throw n=Qo,Or(t,0),Yi(t,e),xn(t,At()),n;if(n===6)throw Error(fe(345));return t.finishedWork=t.current.alternate,t.finishedLanes=e,Rr(t,fn,bi),xn(t,At()),null}function Cp(t,e){var n=st;st|=1;try{return t(e)}finally{st=n,st===0&&(Gs=At()+500,_c&&xr())}}function Wr(t){Qi!==null&&Qi.tag===0&&!(st&6)&&Ns();var e=st;st|=1;var n=Nn.transition,i=ct;try{if(Nn.transition=null,ct=1,t)return t()}finally{ct=i,Nn.transition=n,st=e,!(st&6)&&xr()}}function Ap(){Sn=Cs.current,vt(Cs)}function Or(t,e){t.finishedWork=null,t.finishedLanes=0;var n=t.timeoutHandle;if(n!==-1&&(t.timeoutHandle=-1,P_(n)),It!==null)for(n=It.return;n!==null;){var i=n;switch(lp(i),i.tag){case 1:i=i.type.childContextTypes,i!=null&&Hl();break;case 3:Vs(),vt(mn),vt(nn),xp();break;case 5:gp(i);break;case 4:Vs();break;case 13:vt(St);break;case 19:vt(St);break;case 10:fp(i.type._context);break;case 22:case 23:Ap()}n=n.return}if(Ot=t,It=t=lr(t.current,null),Ht=Sn=e,Dt=0,Qo=null,Tp=bc=Gr=0,fn=No=null,Fr!==null){for(e=0;e<Fr.length;e++)if(n=Fr[e],i=n.interleaved,i!==null){n.interleaved=null;var r=i.next,s=n.pending;if(s!==null){var o=s.next;s.next=r,i.next=o}n.pending=i}Fr=null}return t}function w1(t,e){do{var n=It;try{if(dp(),yl.current=Jl,Zl){for(var i=Mt.memoizedState;i!==null;){var r=i.queue;r!==null&&(r.pending=null),i=i.next}Zl=!1}if(Hr=0,kt=Pt=Mt=null,Lo=!1,Yo=0,Ep.current=null,n===null||n.return===null){Dt=1,Qo=e,It=null;break}e:{var s=t,o=n.return,a=n,c=e;if(e=Ht,a.flags|=32768,c!==null&&typeof c=="object"&&typeof c.then=="function"){var u=c,p=a,h=p.tag;if(!(p.mode&1)&&(h===0||h===11||h===15)){var f=p.alternate;f?(p.updateQueue=f.updateQueue,p.memoizedState=f.memoizedState,p.lanes=f.lanes):(p.updateQueue=null,p.memoizedState=null)}var g=tm(o);if(g!==null){g.flags&=-257,nm(g,o,a,s,e),g.mode&1&&em(s,u,e),e=g,c=u;var x=e.updateQueue;if(x===null){var M=new Set;M.add(c),e.updateQueue=M}else x.add(c);break e}else{if(!(e&1)){em(s,u,e),Rp();break e}c=Error(fe(426))}}else if(yt&&a.mode&1){var v=tm(o);if(v!==null){!(v.flags&65536)&&(v.flags|=256),nm(v,o,a,s,e),cp(Hs(c,a));break e}}s=c=Hs(c,a),Dt!==4&&(Dt=2),No===null?No=[s]:No.push(s),s=o;do{switch(s.tag){case 3:s.flags|=65536,e&=-e,s.lanes|=e;var d=c1(s,c,e);$h(s,d);break e;case 1:a=c;var m=s.type,_=s.stateNode;if(!(s.flags&128)&&(typeof m.getDerivedStateFromError=="function"||_!==null&&typeof _.componentDidCatch=="function"&&(or===null||!or.has(_)))){s.flags|=65536,e&=-e,s.lanes|=e;var b=u1(s,a,e);$h(s,b);break e}}s=s.return}while(s!==null)}R1(n)}catch(w){e=w,It===n&&n!==null&&(It=n=n.return);continue}break}while(!0)}function C1(){var t=Ql.current;return Ql.current=Jl,t===null?Jl:t}function Rp(){(Dt===0||Dt===3||Dt===2)&&(Dt=4),Ot===null||!(Gr&268435455)&&!(bc&268435455)||Yi(Ot,Ht)}function nc(t,e){var n=st;st|=2;var i=C1();(Ot!==t||Ht!==e)&&(bi=null,Or(t,e));do try{ny();break}catch(r){w1(t,r)}while(!0);if(dp(),st=n,Ql.current=i,It!==null)throw Error(fe(261));return Ot=null,Ht=0,Dt}function ny(){for(;It!==null;)A1(It)}function iy(){for(;It!==null&&!Av();)A1(It)}function A1(t){var e=P1(t.alternate,t,Sn);t.memoizedProps=t.pendingProps,e===null?R1(t):It=e,Ep.current=null}function R1(t){var e=t;do{var n=e.alternate;if(t=e.return,e.flags&32768){if(n=Y_(n,e),n!==null){n.flags&=32767,It=n;return}if(t!==null)t.flags|=32768,t.subtreeFlags=0,t.deletions=null;else{Dt=6,It=null;return}}else if(n=K_(n,e,Sn),n!==null){It=n;return}if(e=e.sibling,e!==null){It=e;return}It=e=t}while(e!==null);Dt===0&&(Dt=5)}function Rr(t,e,n){var i=ct,r=Nn.transition;try{Nn.transition=null,ct=1,ry(t,e,n,i)}finally{Nn.transition=r,ct=i}return null}function ry(t,e,n,i){do Ns();while(Qi!==null);if(st&6)throw Error(fe(327));n=t.finishedWork;var r=t.finishedLanes;if(n===null)return null;if(t.finishedWork=null,t.finishedLanes=0,n===t.current)throw Error(fe(177));t.callbackNode=null,t.callbackPriority=0;var s=n.lanes|n.childLanes;if(Ov(t,s),t===Ot&&(It=Ot=null,Ht=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||Na||(Na=!0,D1(kl,function(){return Ns(),null})),s=(n.flags&15990)!==0,n.subtreeFlags&15990||s){s=Nn.transition,Nn.transition=null;var o=ct;ct=1;var a=st;st|=4,Ep.current=null,J_(t,n),b1(n,t),E_(gd),zl=!!md,gd=md=null,t.current=n,Q_(n),Rv(),st=a,ct=o,Nn.transition=s}else t.current=n;if(Na&&(Na=!1,Qi=t,tc=r),s=t.pendingLanes,s===0&&(or=null),Dv(n.stateNode),xn(t,At()),e!==null)for(i=t.onRecoverableError,n=0;n<e.length;n++)r=e[n],i(r.value,{componentStack:r.stack,digest:r.digest});if(ec)throw ec=!1,t=Ud,Ud=null,t;return tc&1&&t.tag!==0&&Ns(),s=t.pendingLanes,s&1?t===kd?Uo++:(Uo=0,kd=t):Uo=0,xr(),null}function Ns(){if(Qi!==null){var t=cg(tc),e=Nn.transition,n=ct;try{if(Nn.transition=null,ct=16>t?16:t,Qi===null)var i=!1;else{if(t=Qi,Qi=null,tc=0,st&6)throw Error(fe(331));var r=st;for(st|=4,Ce=t.current;Ce!==null;){var s=Ce,o=s.child;if(Ce.flags&16){var a=s.deletions;if(a!==null){for(var c=0;c<a.length;c++){var u=a[c];for(Ce=u;Ce!==null;){var p=Ce;switch(p.tag){case 0:case 11:case 15:Fo(8,p,s)}var h=p.child;if(h!==null)h.return=p,Ce=h;else for(;Ce!==null;){p=Ce;var f=p.sibling,g=p.return;if(y1(p),p===u){Ce=null;break}if(f!==null){f.return=g,Ce=f;break}Ce=g}}}var x=s.alternate;if(x!==null){var M=x.child;if(M!==null){x.child=null;do{var v=M.sibling;M.sibling=null,M=v}while(M!==null)}}Ce=s}}if(s.subtreeFlags&2064&&o!==null)o.return=s,Ce=o;else e:for(;Ce!==null;){if(s=Ce,s.flags&2048)switch(s.tag){case 0:case 11:case 15:Fo(9,s,s.return)}var d=s.sibling;if(d!==null){d.return=s.return,Ce=d;break e}Ce=s.return}}var m=t.current;for(Ce=m;Ce!==null;){o=Ce;var _=o.child;if(o.subtreeFlags&2064&&_!==null)_.return=o,Ce=_;else e:for(o=m;Ce!==null;){if(a=Ce,a.flags&2048)try{switch(a.tag){case 0:case 11:case 15:Mc(9,a)}}catch(w){Tt(a,a.return,w)}if(a===o){Ce=null;break e}var b=a.sibling;if(b!==null){b.return=a.return,Ce=b;break e}Ce=a.return}}if(st=r,xr(),li&&typeof li.onPostCommitFiberRoot=="function")try{li.onPostCommitFiberRoot(hc,t)}catch{}i=!0}return i}finally{ct=n,Nn.transition=e}}return!1}function mm(t,e,n){e=Hs(n,e),e=c1(t,e,1),t=sr(t,e,1),e=cn(),t!==null&&(aa(t,1,e),xn(t,e))}function Tt(t,e,n){if(t.tag===3)mm(t,t,n);else for(;e!==null;){if(e.tag===3){mm(e,t,n);break}else if(e.tag===1){var i=e.stateNode;if(typeof e.type.getDerivedStateFromError=="function"||typeof i.componentDidCatch=="function"&&(or===null||!or.has(i))){t=Hs(n,t),t=u1(e,t,1),e=sr(e,t,1),t=cn(),e!==null&&(aa(e,1,t),xn(e,t));break}}e=e.return}}function sy(t,e,n){var i=t.pingCache;i!==null&&i.delete(e),e=cn(),t.pingedLanes|=t.suspendedLanes&n,Ot===t&&(Ht&n)===n&&(Dt===4||Dt===3&&(Ht&130023424)===Ht&&500>At()-wp?Or(t,0):Tp|=n),xn(t,e)}function I1(t,e){e===0&&(t.mode&1?(e=Ta,Ta<<=1,!(Ta&130023424)&&(Ta=4194304)):e=1);var n=cn();t=Li(t,e),t!==null&&(aa(t,e,n),xn(t,n))}function oy(t){var e=t.memoizedState,n=0;e!==null&&(n=e.retryLane),I1(t,n)}function ay(t,e){var n=0;switch(t.tag){case 13:var i=t.stateNode,r=t.memoizedState;r!==null&&(n=r.retryLane);break;case 19:i=t.stateNode;break;default:throw Error(fe(314))}i!==null&&i.delete(e),I1(t,n)}var P1;P1=function(t,e,n){if(t!==null)if(t.memoizedProps!==e.pendingProps||mn.current)hn=!0;else{if(!(t.lanes&n)&&!(e.flags&128))return hn=!1,$_(t,e,n);hn=!!(t.flags&131072)}else hn=!1,yt&&e.flags&1048576&&Ng(e,Xl,e.index);switch(e.lanes=0,e.tag){case 2:var i=e.type;Ml(t,e),t=e.pendingProps;var r=zs(e,nn.current);Fs(e,n),r=_p(null,e,i,t,r,n);var s=yp();return e.flags|=1,typeof r=="object"&&r!==null&&typeof r.render=="function"&&r.$$typeof===void 0?(e.tag=1,e.memoizedState=null,e.updateQueue=null,gn(i)?(s=!0,Gl(e)):s=!1,e.memoizedState=r.state!==null&&r.state!==void 0?r.state:null,hp(e),r.updater=Sc,e.stateNode=r,r._reactInternals=e,Td(e,i,t,n),e=Ad(null,e,i,!0,s,n)):(e.tag=0,yt&&s&&ap(e),an(null,e,r,n),e=e.child),e;case 16:i=e.elementType;e:{switch(Ml(t,e),t=e.pendingProps,r=i._init,i=r(i._payload),e.type=i,r=e.tag=cy(i),t=Hn(i,t),r){case 0:e=Cd(null,e,i,t,n);break e;case 1:e=sm(null,e,i,t,n);break e;case 11:e=im(null,e,i,t,n);break e;case 14:e=rm(null,e,i,Hn(i.type,t),n);break e}throw Error(fe(306,i,""))}return e;case 0:return i=e.type,r=e.pendingProps,r=e.elementType===i?r:Hn(i,r),Cd(t,e,i,r,n);case 1:return i=e.type,r=e.pendingProps,r=e.elementType===i?r:Hn(i,r),sm(t,e,i,r,n);case 3:e:{if(h1(e),t===null)throw Error(fe(387));i=e.pendingProps,s=e.memoizedState,r=s.element,jg(t,e),Kl(e,i,null,n);var o=e.memoizedState;if(i=o.element,s.isDehydrated)if(s={element:i,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},e.updateQueue.baseState=s,e.memoizedState=s,e.flags&256){r=Hs(Error(fe(423)),e),e=om(t,e,i,n,r);break e}else if(i!==r){r=Hs(Error(fe(424)),e),e=om(t,e,i,n,r);break e}else for(En=rr(e.stateNode.containerInfo.firstChild),Tn=e,yt=!0,Wn=null,n=zg(e,null,i,n),e.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Bs(),i===r){e=Fi(t,e,n);break e}an(t,e,i,n)}e=e.child}return e;case 5:return Vg(e),t===null&&Md(e),i=e.type,r=e.pendingProps,s=t!==null?t.memoizedProps:null,o=r.children,xd(i,r)?o=null:s!==null&&xd(i,s)&&(e.flags|=32),p1(t,e),an(t,e,o,n),e.child;case 6:return t===null&&Md(e),null;case 13:return m1(t,e,n);case 4:return mp(e,e.stateNode.containerInfo),i=e.pendingProps,t===null?e.child=js(e,null,i,n):an(t,e,i,n),e.child;case 11:return i=e.type,r=e.pendingProps,r=e.elementType===i?r:Hn(i,r),im(t,e,i,r,n);case 7:return an(t,e,e.pendingProps,n),e.child;case 8:return an(t,e,e.pendingProps.children,n),e.child;case 12:return an(t,e,e.pendingProps.children,n),e.child;case 10:e:{if(i=e.type._context,r=e.pendingProps,s=e.memoizedProps,o=r.value,gt(ql,i._currentValue),i._currentValue=o,s!==null)if(Yn(s.value,o)){if(s.children===r.children&&!mn.current){e=Fi(t,e,n);break e}}else for(s=e.child,s!==null&&(s.return=e);s!==null;){var a=s.dependencies;if(a!==null){o=s.child;for(var c=a.firstContext;c!==null;){if(c.context===i){if(s.tag===1){c=Ai(-1,n&-n),c.tag=2;var u=s.updateQueue;if(u!==null){u=u.shared;var p=u.pending;p===null?c.next=c:(c.next=p.next,p.next=c),u.pending=c}}s.lanes|=n,c=s.alternate,c!==null&&(c.lanes|=n),bd(s.return,n,e),a.lanes|=n;break}c=c.next}}else if(s.tag===10)o=s.type===e.type?null:s.child;else if(s.tag===18){if(o=s.return,o===null)throw Error(fe(341));o.lanes|=n,a=o.alternate,a!==null&&(a.lanes|=n),bd(o,n,e),o=s.sibling}else o=s.child;if(o!==null)o.return=s;else for(o=s;o!==null;){if(o===e){o=null;break}if(s=o.sibling,s!==null){s.return=o.return,o=s;break}o=o.return}s=o}an(t,e,r.children,n),e=e.child}return e;case 9:return r=e.type,i=e.pendingProps.children,Fs(e,n),r=Un(r),i=i(r),e.flags|=1,an(t,e,i,n),e.child;case 14:return i=e.type,r=Hn(i,e.pendingProps),r=Hn(i.type,r),rm(t,e,i,r,n);case 15:return d1(t,e,e.type,e.pendingProps,n);case 17:return i=e.type,r=e.pendingProps,r=e.elementType===i?r:Hn(i,r),Ml(t,e),e.tag=1,gn(i)?(t=!0,Gl(e)):t=!1,Fs(e,n),l1(e,i,r),Td(e,i,r,n),Ad(null,e,i,!0,t,n);case 19:return g1(t,e,n);case 22:return f1(t,e,n)}throw Error(fe(156,e.tag))};function D1(t,e){return sg(t,e)}function ly(t,e,n,i){this.tag=t,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=e,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=i,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Fn(t,e,n,i){return new ly(t,e,n,i)}function Ip(t){return t=t.prototype,!(!t||!t.isReactComponent)}function cy(t){if(typeof t=="function")return Ip(t)?1:0;if(t!=null){if(t=t.$$typeof,t===Kf)return 11;if(t===Yf)return 14}return 2}function lr(t,e){var n=t.alternate;return n===null?(n=Fn(t.tag,e,t.key,t.mode),n.elementType=t.elementType,n.type=t.type,n.stateNode=t.stateNode,n.alternate=t,t.alternate=n):(n.pendingProps=e,n.type=t.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=t.flags&14680064,n.childLanes=t.childLanes,n.lanes=t.lanes,n.child=t.child,n.memoizedProps=t.memoizedProps,n.memoizedState=t.memoizedState,n.updateQueue=t.updateQueue,e=t.dependencies,n.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext},n.sibling=t.sibling,n.index=t.index,n.ref=t.ref,n}function Tl(t,e,n,i,r,s){var o=2;if(i=t,typeof t=="function")Ip(t)&&(o=1);else if(typeof t=="string")o=5;else e:switch(t){case xs:return zr(n.children,r,s,e);case $f:o=8,r|=8;break;case Ku:return t=Fn(12,n,e,r|2),t.elementType=Ku,t.lanes=s,t;case Yu:return t=Fn(13,n,e,r),t.elementType=Yu,t.lanes=s,t;case Zu:return t=Fn(19,n,e,r),t.elementType=Zu,t.lanes=s,t;case V0:return Ec(n,r,s,e);default:if(typeof t=="object"&&t!==null)switch(t.$$typeof){case B0:o=10;break e;case j0:o=9;break e;case Kf:o=11;break e;case Yf:o=14;break e;case qi:o=16,i=null;break e}throw Error(fe(130,t==null?t:typeof t,""))}return e=Fn(o,n,e,r),e.elementType=t,e.type=i,e.lanes=s,e}function zr(t,e,n,i){return t=Fn(7,t,i,e),t.lanes=n,t}function Ec(t,e,n,i){return t=Fn(22,t,i,e),t.elementType=V0,t.lanes=n,t.stateNode={isHidden:!1},t}function cu(t,e,n){return t=Fn(6,t,null,e),t.lanes=n,t}function uu(t,e,n){return e=Fn(4,t.children!==null?t.children:[],t.key,e),e.lanes=n,e.stateNode={containerInfo:t.containerInfo,pendingChildren:null,implementation:t.implementation},e}function uy(t,e,n,i,r){this.tag=e,this.containerInfo=t,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Gc(0),this.expirationTimes=Gc(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Gc(0),this.identifierPrefix=i,this.onRecoverableError=r,this.mutableSourceEagerHydrationData=null}function Pp(t,e,n,i,r,s,o,a,c){return t=new uy(t,e,n,a,c),e===1?(e=1,s===!0&&(e|=8)):e=0,s=Fn(3,null,null,e),t.current=s,s.stateNode=t,s.memoizedState={element:i,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},hp(s),t}function dy(t,e,n){var i=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:gs,key:i==null?null:""+i,children:t,containerInfo:e,implementation:n}}function L1(t){if(!t)return fr;t=t._reactInternals;e:{if($r(t)!==t||t.tag!==1)throw Error(fe(170));var e=t;do{switch(e.tag){case 3:e=e.stateNode.context;break e;case 1:if(gn(e.type)){e=e.stateNode.__reactInternalMemoizedMergedChildContext;break e}}e=e.return}while(e!==null);throw Error(fe(171))}if(t.tag===1){var n=t.type;if(gn(n))return Lg(t,n,e)}return e}function F1(t,e,n,i,r,s,o,a,c){return t=Pp(n,i,!0,t,r,s,o,a,c),t.context=L1(null),n=t.current,i=cn(),r=ar(n),s=Ai(i,r),s.callback=e??null,sr(n,s,r),t.current.lanes=r,aa(t,r,i),xn(t,i),t}function Tc(t,e,n,i){var r=e.current,s=cn(),o=ar(r);return n=L1(n),e.context===null?e.context=n:e.pendingContext=n,e=Ai(s,o),e.payload={element:t},i=i===void 0?null:i,i!==null&&(e.callback=i),t=sr(r,e,o),t!==null&&($n(t,r,o,s),_l(t,r,o)),o}function ic(t){if(t=t.current,!t.child)return null;switch(t.child.tag){case 5:return t.child.stateNode;default:return t.child.stateNode}}function gm(t,e){if(t=t.memoizedState,t!==null&&t.dehydrated!==null){var n=t.retryLane;t.retryLane=n!==0&&n<e?n:e}}function Dp(t,e){gm(t,e),(t=t.alternate)&&gm(t,e)}function fy(){return null}var N1=typeof reportError=="function"?reportError:function(t){console.error(t)};function Lp(t){this._internalRoot=t}wc.prototype.render=Lp.prototype.render=function(t){var e=this._internalRoot;if(e===null)throw Error(fe(409));Tc(t,e,null,null)};wc.prototype.unmount=Lp.prototype.unmount=function(){var t=this._internalRoot;if(t!==null){this._internalRoot=null;var e=t.containerInfo;Wr(function(){Tc(null,t,null,null)}),e[Di]=null}};function wc(t){this._internalRoot=t}wc.prototype.unstable_scheduleHydration=function(t){if(t){var e=fg();t={blockedOn:null,target:t,priority:e};for(var n=0;n<Ki.length&&e!==0&&e<Ki[n].priority;n++);Ki.splice(n,0,t),n===0&&hg(t)}};function Fp(t){return!(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)}function Cc(t){return!(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11&&(t.nodeType!==8||t.nodeValue!==" react-mount-point-unstable "))}function xm(){}function py(t,e,n,i,r){if(r){if(typeof i=="function"){var s=i;i=function(){var u=ic(o);s.call(u)}}var o=F1(e,i,t,0,null,!1,!1,"",xm);return t._reactRootContainer=o,t[Di]=o.current,Wo(t.nodeType===8?t.parentNode:t),Wr(),o}for(;r=t.lastChild;)t.removeChild(r);if(typeof i=="function"){var a=i;i=function(){var u=ic(c);a.call(u)}}var c=Pp(t,0,!1,null,null,!1,!1,"",xm);return t._reactRootContainer=c,t[Di]=c.current,Wo(t.nodeType===8?t.parentNode:t),Wr(function(){Tc(e,c,n,i)}),c}function Ac(t,e,n,i,r){var s=n._reactRootContainer;if(s){var o=s;if(typeof r=="function"){var a=r;r=function(){var c=ic(o);a.call(c)}}Tc(e,o,t,r)}else o=py(n,e,t,r,i);return ic(o)}ug=function(t){switch(t.tag){case 3:var e=t.stateNode;if(e.current.memoizedState.isDehydrated){var n=Eo(e.pendingLanes);n!==0&&(Qf(e,n|1),xn(e,At()),!(st&6)&&(Gs=At()+500,xr()))}break;case 13:Wr(function(){var i=Li(t,1);if(i!==null){var r=cn();$n(i,t,1,r)}}),Dp(t,1)}};ep=function(t){if(t.tag===13){var e=Li(t,134217728);if(e!==null){var n=cn();$n(e,t,134217728,n)}Dp(t,134217728)}};dg=function(t){if(t.tag===13){var e=ar(t),n=Li(t,e);if(n!==null){var i=cn();$n(n,t,e,i)}Dp(t,e)}};fg=function(){return ct};pg=function(t,e){var n=ct;try{return ct=t,e()}finally{ct=n}};ad=function(t,e,n){switch(e){case"input":if(ed(t,n),e=n.name,n.type==="radio"&&e!=null){for(n=t;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+e)+'][type="radio"]'),e=0;e<n.length;e++){var i=n[e];if(i!==t&&i.form===t.form){var r=vc(i);if(!r)throw Error(fe(90));G0(i),ed(i,r)}}}break;case"textarea":X0(t,n);break;case"select":e=n.value,e!=null&&Is(t,!!n.multiple,e,!1)}};Q0=Cp;eg=Wr;var hy={usingClientEntryPoint:!1,Events:[ca,Ss,vc,Z0,J0,Cp]},uo={findFiberByHostInstance:Lr,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},my={bundleType:uo.bundleType,version:uo.version,rendererPackageName:uo.rendererPackageName,rendererConfig:uo.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:ki.ReactCurrentDispatcher,findHostInstanceByFiber:function(t){return t=ig(t),t===null?null:t.stateNode},findFiberByHostInstance:uo.findFiberByHostInstance||fy,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Ua=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Ua.isDisabled&&Ua.supportsFiber)try{hc=Ua.inject(my),li=Ua}catch{}}Cn.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=hy;Cn.createPortal=function(t,e){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!Fp(e))throw Error(fe(200));return dy(t,e,null,n)};Cn.createRoot=function(t,e){if(!Fp(t))throw Error(fe(299));var n=!1,i="",r=N1;return e!=null&&(e.unstable_strictMode===!0&&(n=!0),e.identifierPrefix!==void 0&&(i=e.identifierPrefix),e.onRecoverableError!==void 0&&(r=e.onRecoverableError)),e=Pp(t,1,!1,null,null,n,!1,i,r),t[Di]=e.current,Wo(t.nodeType===8?t.parentNode:t),new Lp(e)};Cn.findDOMNode=function(t){if(t==null)return null;if(t.nodeType===1)return t;var e=t._reactInternals;if(e===void 0)throw typeof t.render=="function"?Error(fe(188)):(t=Object.keys(t).join(","),Error(fe(268,t)));return t=ig(e),t=t===null?null:t.stateNode,t};Cn.flushSync=function(t){return Wr(t)};Cn.hydrate=function(t,e,n){if(!Cc(e))throw Error(fe(200));return Ac(null,t,e,!0,n)};Cn.hydrateRoot=function(t,e,n){if(!Fp(t))throw Error(fe(405));var i=n!=null&&n.hydratedSources||null,r=!1,s="",o=N1;if(n!=null&&(n.unstable_strictMode===!0&&(r=!0),n.identifierPrefix!==void 0&&(s=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),e=F1(e,null,t,1,n??null,r,!1,s,o),t[Di]=e.current,Wo(t),i)for(t=0;t<i.length;t++)n=i[t],r=n._getVersion,r=r(n._source),e.mutableSourceEagerHydrationData==null?e.mutableSourceEagerHydrationData=[n,r]:e.mutableSourceEagerHydrationData.push(n,r);return new wc(e)};Cn.render=function(t,e,n){if(!Cc(e))throw Error(fe(200));return Ac(null,t,e,!1,n)};Cn.unmountComponentAtNode=function(t){if(!Cc(t))throw Error(fe(40));return t._reactRootContainer?(Wr(function(){Ac(null,null,t,!1,function(){t._reactRootContainer=null,t[Di]=null})}),!0):!1};Cn.unstable_batchedUpdates=Cp;Cn.unstable_renderSubtreeIntoContainer=function(t,e,n,i){if(!Cc(n))throw Error(fe(200));if(t==null||t._reactInternals===void 0)throw Error(fe(38));return Ac(t,e,n,!1,i)};Cn.version="18.3.1-next-f1338f8080-20240426";function U1(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(U1)}catch(t){console.error(t)}}U1(),U0.exports=Cn;var da=U0.exports,vm=da;qu.createRoot=vm.createRoot,qu.hydrateRoot=vm.hydrateRoot;/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const Np="183",gy=0,_m=1,xy=2,wl=1,vy=2,wo=3,pr=0,en=1,ri=2,Ri=0,Us=1,Bd=2,ym=3,Sm=4,_y=5,Pr=100,yy=101,Sy=102,My=103,by=104,Ey=200,Ty=201,wy=202,Cy=203,jd=204,Vd=205,Ay=206,Ry=207,Iy=208,Py=209,Dy=210,Ly=211,Fy=212,Ny=213,Uy=214,Hd=0,Gd=1,Wd=2,Ws=3,Xd=4,qd=5,$d=6,Kd=7,k1=0,ky=1,Oy=2,ui=0,O1=1,z1=2,B1=3,j1=4,V1=5,H1=6,G1=7,W1=300,Xr=301,Xs=302,du=303,fu=304,Rc=306,Yd=1e3,Ci=1001,Zd=1002,Vt=1003,zy=1004,ka=1005,Qt=1006,pu=1007,Ur=1008,bn=1009,X1=1010,q1=1011,ea=1012,Up=1013,fi=1014,oi=1015,Ni=1016,kp=1017,Op=1018,ta=1020,$1=35902,K1=35899,Y1=1021,Z1=1022,Xn=1023,Ui=1026,kr=1027,J1=1028,zp=1029,qs=1030,Bp=1031,jp=1033,Cl=33776,Al=33777,Rl=33778,Il=33779,Jd=35840,Qd=35841,ef=35842,tf=35843,nf=36196,rf=37492,sf=37496,of=37488,af=37489,lf=37490,cf=37491,uf=37808,df=37809,ff=37810,pf=37811,hf=37812,mf=37813,gf=37814,xf=37815,vf=37816,_f=37817,yf=37818,Sf=37819,Mf=37820,bf=37821,Ef=36492,Tf=36494,wf=36495,Cf=36283,Af=36284,Rf=36285,If=36286,By=3200,Q1=0,jy=1,Zi="",In="srgb",$s="srgb-linear",rc="linear",lt="srgb",Jr=7680,Mm=519,Vy=512,Hy=513,Gy=514,Vp=515,Wy=516,Xy=517,Hp=518,qy=519,Pf=35044,bm="300 es",ai=2e3,na=2001;function $y(t){for(let e=t.length-1;e>=0;--e)if(t[e]>=65535)return!0;return!1}function sc(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function Ky(){const t=sc("canvas");return t.style.display="block",t}const Em={};function oc(...t){const e="THREE."+t.shift();console.log(e,...t)}function ex(t){const e=t[0];if(typeof e=="string"&&e.startsWith("TSL:")){const n=t[1];n&&n.isStackTrace?t[0]+=" "+n.getLocation():t[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return t}function He(...t){t=ex(t);const e="THREE."+t.shift();{const n=t[0];n&&n.isStackTrace?console.warn(n.getError(e)):console.warn(e,...t)}}function it(...t){t=ex(t);const e="THREE."+t.shift();{const n=t[0];n&&n.isStackTrace?console.error(n.getError(e)):console.error(e,...t)}}function ac(...t){const e=t.join(" ");e in Em||(Em[e]=!0,He(...t))}function Yy(t,e,n){return new Promise(function(i,r){function s(){switch(t.clientWaitSync(e,t.SYNC_FLUSH_COMMANDS_BIT,0)){case t.WAIT_FAILED:r();break;case t.TIMEOUT_EXPIRED:setTimeout(s,n);break;default:i()}}setTimeout(s,n)})}const Zy={[Hd]:Gd,[Wd]:$d,[Xd]:Kd,[Ws]:qd,[Gd]:Hd,[$d]:Wd,[Kd]:Xd,[qd]:Ws};class eo{addEventListener(e,n){this._listeners===void 0&&(this._listeners={});const i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(n)===-1&&i[e].push(n)}hasEventListener(e,n){const i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(n)!==-1}removeEventListener(e,n){const i=this._listeners;if(i===void 0)return;const r=i[e];if(r!==void 0){const s=r.indexOf(n);s!==-1&&r.splice(s,1)}}dispatchEvent(e){const n=this._listeners;if(n===void 0)return;const i=n[e.type];if(i!==void 0){e.target=this;const r=i.slice(0);for(let s=0,o=r.length;s<o;s++)r[s].call(this,e);e.target=null}}}const Yt=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],hu=Math.PI/180,Df=180/Math.PI;function cr(){const t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(Yt[t&255]+Yt[t>>8&255]+Yt[t>>16&255]+Yt[t>>24&255]+"-"+Yt[e&255]+Yt[e>>8&255]+"-"+Yt[e>>16&15|64]+Yt[e>>24&255]+"-"+Yt[n&63|128]+Yt[n>>8&255]+"-"+Yt[n>>16&255]+Yt[n>>24&255]+Yt[i&255]+Yt[i>>8&255]+Yt[i>>16&255]+Yt[i>>24&255]).toLowerCase()}function nt(t,e,n){return Math.max(e,Math.min(n,t))}function Jy(t,e){return(t%e+e)%e}function mu(t,e,n){return(1-n)*t+n*e}function si(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return t/4294967295;case Uint16Array:return t/65535;case Uint8Array:return t/255;case Int32Array:return Math.max(t/2147483647,-1);case Int16Array:return Math.max(t/32767,-1);case Int8Array:return Math.max(t/127,-1);default:throw new Error("Invalid component type.")}}function ft(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return Math.round(t*4294967295);case Uint16Array:return Math.round(t*65535);case Uint8Array:return Math.round(t*255);case Int32Array:return Math.round(t*2147483647);case Int16Array:return Math.round(t*32767);case Int8Array:return Math.round(t*127);default:throw new Error("Invalid component type.")}}class Ze{constructor(e=0,n=0){Ze.prototype.isVector2=!0,this.x=e,this.y=n}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,n){return this.x=e,this.y=n,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const n=this.x,i=this.y,r=e.elements;return this.x=r[0]*n+r[3]*i+r[6],this.y=r[1]*n+r[4]*i+r[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,n){return this.x=nt(this.x,e.x,n.x),this.y=nt(this.y,e.y,n.y),this}clampScalar(e,n){return this.x=nt(this.x,e,n),this.y=nt(this.y,e,n),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(nt(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;const i=this.dot(e)/n;return Math.acos(nt(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const n=this.x-e.x,i=this.y-e.y;return n*n+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this}rotateAround(e,n){const i=Math.cos(n),r=Math.sin(n),s=this.x-e.x,o=this.y-e.y;return this.x=s*i-o*r+e.x,this.y=s*r+o*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class to{constructor(e=0,n=0,i=0,r=1){this.isQuaternion=!0,this._x=e,this._y=n,this._z=i,this._w=r}static slerpFlat(e,n,i,r,s,o,a){let c=i[r+0],u=i[r+1],p=i[r+2],h=i[r+3],f=s[o+0],g=s[o+1],x=s[o+2],M=s[o+3];if(h!==M||c!==f||u!==g||p!==x){let v=c*f+u*g+p*x+h*M;v<0&&(f=-f,g=-g,x=-x,M=-M,v=-v);let d=1-a;if(v<.9995){const m=Math.acos(v),_=Math.sin(m);d=Math.sin(d*m)/_,a=Math.sin(a*m)/_,c=c*d+f*a,u=u*d+g*a,p=p*d+x*a,h=h*d+M*a}else{c=c*d+f*a,u=u*d+g*a,p=p*d+x*a,h=h*d+M*a;const m=1/Math.sqrt(c*c+u*u+p*p+h*h);c*=m,u*=m,p*=m,h*=m}}e[n]=c,e[n+1]=u,e[n+2]=p,e[n+3]=h}static multiplyQuaternionsFlat(e,n,i,r,s,o){const a=i[r],c=i[r+1],u=i[r+2],p=i[r+3],h=s[o],f=s[o+1],g=s[o+2],x=s[o+3];return e[n]=a*x+p*h+c*g-u*f,e[n+1]=c*x+p*f+u*h-a*g,e[n+2]=u*x+p*g+a*f-c*h,e[n+3]=p*x-a*h-c*f-u*g,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,n,i,r){return this._x=e,this._y=n,this._z=i,this._w=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,n=!0){const i=e._x,r=e._y,s=e._z,o=e._order,a=Math.cos,c=Math.sin,u=a(i/2),p=a(r/2),h=a(s/2),f=c(i/2),g=c(r/2),x=c(s/2);switch(o){case"XYZ":this._x=f*p*h+u*g*x,this._y=u*g*h-f*p*x,this._z=u*p*x+f*g*h,this._w=u*p*h-f*g*x;break;case"YXZ":this._x=f*p*h+u*g*x,this._y=u*g*h-f*p*x,this._z=u*p*x-f*g*h,this._w=u*p*h+f*g*x;break;case"ZXY":this._x=f*p*h-u*g*x,this._y=u*g*h+f*p*x,this._z=u*p*x+f*g*h,this._w=u*p*h-f*g*x;break;case"ZYX":this._x=f*p*h-u*g*x,this._y=u*g*h+f*p*x,this._z=u*p*x-f*g*h,this._w=u*p*h+f*g*x;break;case"YZX":this._x=f*p*h+u*g*x,this._y=u*g*h+f*p*x,this._z=u*p*x-f*g*h,this._w=u*p*h-f*g*x;break;case"XZY":this._x=f*p*h-u*g*x,this._y=u*g*h-f*p*x,this._z=u*p*x+f*g*h,this._w=u*p*h+f*g*x;break;default:He("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return n===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,n){const i=n/2,r=Math.sin(i);return this._x=e.x*r,this._y=e.y*r,this._z=e.z*r,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){const n=e.elements,i=n[0],r=n[4],s=n[8],o=n[1],a=n[5],c=n[9],u=n[2],p=n[6],h=n[10],f=i+a+h;if(f>0){const g=.5/Math.sqrt(f+1);this._w=.25/g,this._x=(p-c)*g,this._y=(s-u)*g,this._z=(o-r)*g}else if(i>a&&i>h){const g=2*Math.sqrt(1+i-a-h);this._w=(p-c)/g,this._x=.25*g,this._y=(r+o)/g,this._z=(s+u)/g}else if(a>h){const g=2*Math.sqrt(1+a-i-h);this._w=(s-u)/g,this._x=(r+o)/g,this._y=.25*g,this._z=(c+p)/g}else{const g=2*Math.sqrt(1+h-i-a);this._w=(o-r)/g,this._x=(s+u)/g,this._y=(c+p)/g,this._z=.25*g}return this._onChangeCallback(),this}setFromUnitVectors(e,n){let i=e.dot(n)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*n.z-e.z*n.y,this._y=e.z*n.x-e.x*n.z,this._z=e.x*n.y-e.y*n.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(nt(this.dot(e),-1,1)))}rotateTowards(e,n){const i=this.angleTo(e);if(i===0)return this;const r=Math.min(1,n/i);return this.slerp(e,r),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,n){const i=e._x,r=e._y,s=e._z,o=e._w,a=n._x,c=n._y,u=n._z,p=n._w;return this._x=i*p+o*a+r*u-s*c,this._y=r*p+o*c+s*a-i*u,this._z=s*p+o*u+i*c-r*a,this._w=o*p-i*a-r*c-s*u,this._onChangeCallback(),this}slerp(e,n){let i=e._x,r=e._y,s=e._z,o=e._w,a=this.dot(e);a<0&&(i=-i,r=-r,s=-s,o=-o,a=-a);let c=1-n;if(a<.9995){const u=Math.acos(a),p=Math.sin(u);c=Math.sin(c*u)/p,n=Math.sin(n*u)/p,this._x=this._x*c+i*n,this._y=this._y*c+r*n,this._z=this._z*c+s*n,this._w=this._w*c+o*n,this._onChangeCallback()}else this._x=this._x*c+i*n,this._y=this._y*c+r*n,this._z=this._z*c+s*n,this._w=this._w*c+o*n,this.normalize();return this}slerpQuaternions(e,n,i){return this.copy(e).slerp(n,i)}random(){const e=2*Math.PI*Math.random(),n=2*Math.PI*Math.random(),i=Math.random(),r=Math.sqrt(1-i),s=Math.sqrt(i);return this.set(r*Math.sin(e),r*Math.cos(e),s*Math.sin(n),s*Math.cos(n))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,n=0){return this._x=e[n],this._y=e[n+1],this._z=e[n+2],this._w=e[n+3],this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._w,e}fromBufferAttribute(e,n){return this._x=e.getX(n),this._y=e.getY(n),this._z=e.getZ(n),this._w=e.getW(n),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class H{constructor(e=0,n=0,i=0){H.prototype.isVector3=!0,this.x=e,this.y=n,this.z=i}set(e,n,i){return i===void 0&&(i=this.z),this.x=e,this.y=n,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,n){return this.x=e.x*n.x,this.y=e.y*n.y,this.z=e.z*n.z,this}applyEuler(e){return this.applyQuaternion(Tm.setFromEuler(e))}applyAxisAngle(e,n){return this.applyQuaternion(Tm.setFromAxisAngle(e,n))}applyMatrix3(e){const n=this.x,i=this.y,r=this.z,s=e.elements;return this.x=s[0]*n+s[3]*i+s[6]*r,this.y=s[1]*n+s[4]*i+s[7]*r,this.z=s[2]*n+s[5]*i+s[8]*r,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const n=this.x,i=this.y,r=this.z,s=e.elements,o=1/(s[3]*n+s[7]*i+s[11]*r+s[15]);return this.x=(s[0]*n+s[4]*i+s[8]*r+s[12])*o,this.y=(s[1]*n+s[5]*i+s[9]*r+s[13])*o,this.z=(s[2]*n+s[6]*i+s[10]*r+s[14])*o,this}applyQuaternion(e){const n=this.x,i=this.y,r=this.z,s=e.x,o=e.y,a=e.z,c=e.w,u=2*(o*r-a*i),p=2*(a*n-s*r),h=2*(s*i-o*n);return this.x=n+c*u+o*h-a*p,this.y=i+c*p+a*u-s*h,this.z=r+c*h+s*p-o*u,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const n=this.x,i=this.y,r=this.z,s=e.elements;return this.x=s[0]*n+s[4]*i+s[8]*r,this.y=s[1]*n+s[5]*i+s[9]*r,this.z=s[2]*n+s[6]*i+s[10]*r,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,n){return this.x=nt(this.x,e.x,n.x),this.y=nt(this.y,e.y,n.y),this.z=nt(this.z,e.z,n.z),this}clampScalar(e,n){return this.x=nt(this.x,e,n),this.y=nt(this.y,e,n),this.z=nt(this.z,e,n),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(nt(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,n){const i=e.x,r=e.y,s=e.z,o=n.x,a=n.y,c=n.z;return this.x=r*c-s*a,this.y=s*o-i*c,this.z=i*a-r*o,this}projectOnVector(e){const n=e.lengthSq();if(n===0)return this.set(0,0,0);const i=e.dot(this)/n;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return gu.copy(this).projectOnVector(e),this.sub(gu)}reflect(e){return this.sub(gu.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;const i=this.dot(e)/n;return Math.acos(nt(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const n=this.x-e.x,i=this.y-e.y,r=this.z-e.z;return n*n+i*i+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,n,i){const r=Math.sin(n)*e;return this.x=r*Math.sin(i),this.y=Math.cos(n)*e,this.z=r*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,n,i){return this.x=e*Math.sin(n),this.y=i,this.z=e*Math.cos(n),this}setFromMatrixPosition(e){const n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this}setFromMatrixScale(e){const n=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),r=this.setFromMatrixColumn(e,2).length();return this.x=n,this.y=i,this.z=r,this}setFromMatrixColumn(e,n){return this.fromArray(e.elements,n*4)}setFromMatrix3Column(e,n){return this.fromArray(e.elements,n*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,n=Math.random()*2-1,i=Math.sqrt(1-n*n);return this.x=i*Math.cos(e),this.y=n,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const gu=new H,Tm=new to;class qe{constructor(e,n,i,r,s,o,a,c,u){qe.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,n,i,r,s,o,a,c,u)}set(e,n,i,r,s,o,a,c,u){const p=this.elements;return p[0]=e,p[1]=r,p[2]=a,p[3]=n,p[4]=s,p[5]=c,p[6]=i,p[7]=o,p[8]=u,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],this}extractBasis(e,n,i){return e.setFromMatrix3Column(this,0),n.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const n=e.elements;return this.set(n[0],n[4],n[8],n[1],n[5],n[9],n[2],n[6],n[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){const i=e.elements,r=n.elements,s=this.elements,o=i[0],a=i[3],c=i[6],u=i[1],p=i[4],h=i[7],f=i[2],g=i[5],x=i[8],M=r[0],v=r[3],d=r[6],m=r[1],_=r[4],b=r[7],w=r[2],A=r[5],E=r[8];return s[0]=o*M+a*m+c*w,s[3]=o*v+a*_+c*A,s[6]=o*d+a*b+c*E,s[1]=u*M+p*m+h*w,s[4]=u*v+p*_+h*A,s[7]=u*d+p*b+h*E,s[2]=f*M+g*m+x*w,s[5]=f*v+g*_+x*A,s[8]=f*d+g*b+x*E,this}multiplyScalar(e){const n=this.elements;return n[0]*=e,n[3]*=e,n[6]*=e,n[1]*=e,n[4]*=e,n[7]*=e,n[2]*=e,n[5]*=e,n[8]*=e,this}determinant(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],c=e[6],u=e[7],p=e[8];return n*o*p-n*a*u-i*s*p+i*a*c+r*s*u-r*o*c}invert(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],c=e[6],u=e[7],p=e[8],h=p*o-a*u,f=a*c-p*s,g=u*s-o*c,x=n*h+i*f+r*g;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const M=1/x;return e[0]=h*M,e[1]=(r*u-p*i)*M,e[2]=(a*i-r*o)*M,e[3]=f*M,e[4]=(p*n-r*c)*M,e[5]=(r*s-a*n)*M,e[6]=g*M,e[7]=(i*c-u*n)*M,e[8]=(o*n-i*s)*M,this}transpose(){let e;const n=this.elements;return e=n[1],n[1]=n[3],n[3]=e,e=n[2],n[2]=n[6],n[6]=e,e=n[5],n[5]=n[7],n[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const n=this.elements;return e[0]=n[0],e[1]=n[3],e[2]=n[6],e[3]=n[1],e[4]=n[4],e[5]=n[7],e[6]=n[2],e[7]=n[5],e[8]=n[8],this}setUvTransform(e,n,i,r,s,o,a){const c=Math.cos(s),u=Math.sin(s);return this.set(i*c,i*u,-i*(c*o+u*a)+o+e,-r*u,r*c,-r*(-u*o+c*a)+a+n,0,0,1),this}scale(e,n){return this.premultiply(xu.makeScale(e,n)),this}rotate(e){return this.premultiply(xu.makeRotation(-e)),this}translate(e,n){return this.premultiply(xu.makeTranslation(e,n)),this}makeTranslation(e,n){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,n,0,0,1),this}makeRotation(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,i,n,0,0,0,1),this}makeScale(e,n){return this.set(e,0,0,0,n,0,0,0,1),this}equals(e){const n=this.elements,i=e.elements;for(let r=0;r<9;r++)if(n[r]!==i[r])return!1;return!0}fromArray(e,n=0){for(let i=0;i<9;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){const i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const xu=new qe,wm=new qe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Cm=new qe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Qy(){const t={enabled:!0,workingColorSpace:$s,spaces:{},convert:function(r,s,o){return this.enabled===!1||s===o||!s||!o||(this.spaces[s].transfer===lt&&(r.r=Ii(r.r),r.g=Ii(r.g),r.b=Ii(r.b)),this.spaces[s].primaries!==this.spaces[o].primaries&&(r.applyMatrix3(this.spaces[s].toXYZ),r.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===lt&&(r.r=ks(r.r),r.g=ks(r.g),r.b=ks(r.b))),r},workingToColorSpace:function(r,s){return this.convert(r,this.workingColorSpace,s)},colorSpaceToWorking:function(r,s){return this.convert(r,s,this.workingColorSpace)},getPrimaries:function(r){return this.spaces[r].primaries},getTransfer:function(r){return r===Zi?rc:this.spaces[r].transfer},getToneMappingMode:function(r){return this.spaces[r].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(r,s=this.workingColorSpace){return r.fromArray(this.spaces[s].luminanceCoefficients)},define:function(r){Object.assign(this.spaces,r)},_getMatrix:function(r,s,o){return r.copy(this.spaces[s].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(r){return this.spaces[r].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(r=this.workingColorSpace){return this.spaces[r].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(r,s){return ac("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),t.workingToColorSpace(r,s)},toWorkingColorSpace:function(r,s){return ac("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),t.colorSpaceToWorking(r,s)}},e=[.64,.33,.3,.6,.15,.06],n=[.2126,.7152,.0722],i=[.3127,.329];return t.define({[$s]:{primaries:e,whitePoint:i,transfer:rc,toXYZ:wm,fromXYZ:Cm,luminanceCoefficients:n,workingColorSpaceConfig:{unpackColorSpace:In},outputColorSpaceConfig:{drawingBufferColorSpace:In}},[In]:{primaries:e,whitePoint:i,transfer:lt,toXYZ:wm,fromXYZ:Cm,luminanceCoefficients:n,outputColorSpaceConfig:{drawingBufferColorSpace:In}}}),t}const rt=Qy();function Ii(t){return t<.04045?t*.0773993808:Math.pow(t*.9478672986+.0521327014,2.4)}function ks(t){return t<.0031308?t*12.92:1.055*Math.pow(t,.41666)-.055}let Qr;class eS{static getDataURL(e,n="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{Qr===void 0&&(Qr=sc("canvas")),Qr.width=e.width,Qr.height=e.height;const r=Qr.getContext("2d");e instanceof ImageData?r.putImageData(e,0,0):r.drawImage(e,0,0,e.width,e.height),i=Qr}return i.toDataURL(n)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const n=sc("canvas");n.width=e.width,n.height=e.height;const i=n.getContext("2d");i.drawImage(e,0,0,e.width,e.height);const r=i.getImageData(0,0,e.width,e.height),s=r.data;for(let o=0;o<s.length;o++)s[o]=Ii(s[o]/255)*255;return i.putImageData(r,0,0),n}else if(e.data){const n=e.data.slice(0);for(let i=0;i<n.length;i++)n instanceof Uint8Array||n instanceof Uint8ClampedArray?n[i]=Math.floor(Ii(n[i]/255)*255):n[i]=Ii(n[i]);return{data:n,width:e.width,height:e.height}}else return He("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let tS=0;class Gp{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:tS++}),this.uuid=cr(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const n=this.data;return typeof HTMLVideoElement<"u"&&n instanceof HTMLVideoElement?e.set(n.videoWidth,n.videoHeight,0):typeof VideoFrame<"u"&&n instanceof VideoFrame?e.set(n.displayHeight,n.displayWidth,0):n!==null?e.set(n.width,n.height,n.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const n=e===void 0||typeof e=="string";if(!n&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const i={uuid:this.uuid,url:""},r=this.data;if(r!==null){let s;if(Array.isArray(r)){s=[];for(let o=0,a=r.length;o<a;o++)r[o].isDataTexture?s.push(vu(r[o].image)):s.push(vu(r[o]))}else s=vu(r);i.url=s}return n||(e.images[this.uuid]=i),i}}function vu(t){return typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap?eS.getDataURL(t):t.data?{data:Array.from(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(He("Texture: Unable to serialize Texture."),{})}let nS=0;const _u=new H;class tn extends eo{constructor(e=tn.DEFAULT_IMAGE,n=tn.DEFAULT_MAPPING,i=Ci,r=Ci,s=Qt,o=Ur,a=Xn,c=bn,u=tn.DEFAULT_ANISOTROPY,p=Zi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:nS++}),this.uuid=cr(),this.name="",this.source=new Gp(e),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=i,this.wrapT=r,this.magFilter=s,this.minFilter=o,this.anisotropy=u,this.format=a,this.internalFormat=null,this.type=c,this.offset=new Ze(0,0),this.repeat=new Ze(1,1),this.center=new Ze(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new qe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=p,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(_u).x}get height(){return this.source.getSize(_u).y}get depth(){return this.source.getSize(_u).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const n in e){const i=e[n];if(i===void 0){He(`Texture.setValues(): parameter '${n}' has value of undefined.`);continue}const r=this[n];if(r===void 0){He(`Texture.setValues(): property '${n}' does not exist.`);continue}r&&i&&r.isVector2&&i.isVector2||r&&i&&r.isVector3&&i.isVector3||r&&i&&r.isMatrix3&&i.isMatrix3?r.copy(i):this[n]=i}}toJSON(e){const n=e===void 0||typeof e=="string";if(!n&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),n||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==W1)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Yd:e.x=e.x-Math.floor(e.x);break;case Ci:e.x=e.x<0?0:1;break;case Zd:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Yd:e.y=e.y-Math.floor(e.y);break;case Ci:e.y=e.y<0?0:1;break;case Zd:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}tn.DEFAULT_IMAGE=null;tn.DEFAULT_MAPPING=W1;tn.DEFAULT_ANISOTROPY=1;class wt{constructor(e=0,n=0,i=0,r=1){wt.prototype.isVector4=!0,this.x=e,this.y=n,this.z=i,this.w=r}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,n,i,r){return this.x=e,this.y=n,this.z=i,this.w=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;case 3:this.w=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this.w=e.w+n.w,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this.w+=e.w*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this.w=e.w-n.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const n=this.x,i=this.y,r=this.z,s=this.w,o=e.elements;return this.x=o[0]*n+o[4]*i+o[8]*r+o[12]*s,this.y=o[1]*n+o[5]*i+o[9]*r+o[13]*s,this.z=o[2]*n+o[6]*i+o[10]*r+o[14]*s,this.w=o[3]*n+o[7]*i+o[11]*r+o[15]*s,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const n=Math.sqrt(1-e.w*e.w);return n<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/n,this.y=e.y/n,this.z=e.z/n),this}setAxisAngleFromRotationMatrix(e){let n,i,r,s;const c=e.elements,u=c[0],p=c[4],h=c[8],f=c[1],g=c[5],x=c[9],M=c[2],v=c[6],d=c[10];if(Math.abs(p-f)<.01&&Math.abs(h-M)<.01&&Math.abs(x-v)<.01){if(Math.abs(p+f)<.1&&Math.abs(h+M)<.1&&Math.abs(x+v)<.1&&Math.abs(u+g+d-3)<.1)return this.set(1,0,0,0),this;n=Math.PI;const _=(u+1)/2,b=(g+1)/2,w=(d+1)/2,A=(p+f)/4,E=(h+M)/4,y=(x+v)/4;return _>b&&_>w?_<.01?(i=0,r=.707106781,s=.707106781):(i=Math.sqrt(_),r=A/i,s=E/i):b>w?b<.01?(i=.707106781,r=0,s=.707106781):(r=Math.sqrt(b),i=A/r,s=y/r):w<.01?(i=.707106781,r=.707106781,s=0):(s=Math.sqrt(w),i=E/s,r=y/s),this.set(i,r,s,n),this}let m=Math.sqrt((v-x)*(v-x)+(h-M)*(h-M)+(f-p)*(f-p));return Math.abs(m)<.001&&(m=1),this.x=(v-x)/m,this.y=(h-M)/m,this.z=(f-p)/m,this.w=Math.acos((u+g+d-1)/2),this}setFromMatrixPosition(e){const n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this.w=n[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,n){return this.x=nt(this.x,e.x,n.x),this.y=nt(this.y,e.y,n.y),this.z=nt(this.z,e.z,n.z),this.w=nt(this.w,e.w,n.w),this}clampScalar(e,n){return this.x=nt(this.x,e,n),this.y=nt(this.y,e,n),this.z=nt(this.z,e,n),this.w=nt(this.w,e,n),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(nt(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this.w+=(e.w-this.w)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this.w=e.w+(n.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this.w=e[n+3],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e[n+3]=this.w,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this.w=e.getW(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class iS extends eo{constructor(e=1,n=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Qt,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},i),this.isRenderTarget=!0,this.width=e,this.height=n,this.depth=i.depth,this.scissor=new wt(0,0,e,n),this.scissorTest=!1,this.viewport=new wt(0,0,e,n),this.textures=[];const r={width:e,height:n,depth:i.depth},s=new tn(r),o=i.count;for(let a=0;a<o;a++)this.textures[a]=s.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview}_setTextureOptions(e={}){const n={minFilter:Qt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(n.mapping=e.mapping),e.wrapS!==void 0&&(n.wrapS=e.wrapS),e.wrapT!==void 0&&(n.wrapT=e.wrapT),e.wrapR!==void 0&&(n.wrapR=e.wrapR),e.magFilter!==void 0&&(n.magFilter=e.magFilter),e.minFilter!==void 0&&(n.minFilter=e.minFilter),e.format!==void 0&&(n.format=e.format),e.type!==void 0&&(n.type=e.type),e.anisotropy!==void 0&&(n.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(n.colorSpace=e.colorSpace),e.flipY!==void 0&&(n.flipY=e.flipY),e.generateMipmaps!==void 0&&(n.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(n.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(n)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,n,i=1){if(this.width!==e||this.height!==n||this.depth!==i){this.width=e,this.height=n,this.depth=i;for(let r=0,s=this.textures.length;r<s;r++)this.textures[r].image.width=e,this.textures[r].image.height=n,this.textures[r].image.depth=i,this.textures[r].isData3DTexture!==!0&&(this.textures[r].isArrayTexture=this.textures[r].image.depth>1);this.dispose()}this.viewport.set(0,0,e,n),this.scissor.set(0,0,e,n)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let n=0,i=e.textures.length;n<i;n++){this.textures[n]=e.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0,this.textures[n].renderTarget=this;const r=Object.assign({},e.textures[n].image);this.textures[n].source=new Gp(r)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class di extends iS{constructor(e=1,n=1,i={}){super(e,n,i),this.isWebGLRenderTarget=!0}}class tx extends tn{constructor(e=null,n=1,i=1,r=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:n,height:i,depth:r},this.magFilter=Vt,this.minFilter=Vt,this.wrapR=Ci,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class rS extends tn{constructor(e=null,n=1,i=1,r=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:n,height:i,depth:r},this.magFilter=Vt,this.minFilter=Vt,this.wrapR=Ci,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class _t{constructor(e,n,i,r,s,o,a,c,u,p,h,f,g,x,M,v){_t.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,n,i,r,s,o,a,c,u,p,h,f,g,x,M,v)}set(e,n,i,r,s,o,a,c,u,p,h,f,g,x,M,v){const d=this.elements;return d[0]=e,d[4]=n,d[8]=i,d[12]=r,d[1]=s,d[5]=o,d[9]=a,d[13]=c,d[2]=u,d[6]=p,d[10]=h,d[14]=f,d[3]=g,d[7]=x,d[11]=M,d[15]=v,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new _t().fromArray(this.elements)}copy(e){const n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],n[9]=i[9],n[10]=i[10],n[11]=i[11],n[12]=i[12],n[13]=i[13],n[14]=i[14],n[15]=i[15],this}copyPosition(e){const n=this.elements,i=e.elements;return n[12]=i[12],n[13]=i[13],n[14]=i[14],this}setFromMatrix3(e){const n=e.elements;return this.set(n[0],n[3],n[6],0,n[1],n[4],n[7],0,n[2],n[5],n[8],0,0,0,0,1),this}extractBasis(e,n,i){return this.determinant()===0?(e.set(1,0,0),n.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),n.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,n,i){return this.set(e.x,n.x,i.x,0,e.y,n.y,i.y,0,e.z,n.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinant()===0)return this.identity();const n=this.elements,i=e.elements,r=1/es.setFromMatrixColumn(e,0).length(),s=1/es.setFromMatrixColumn(e,1).length(),o=1/es.setFromMatrixColumn(e,2).length();return n[0]=i[0]*r,n[1]=i[1]*r,n[2]=i[2]*r,n[3]=0,n[4]=i[4]*s,n[5]=i[5]*s,n[6]=i[6]*s,n[7]=0,n[8]=i[8]*o,n[9]=i[9]*o,n[10]=i[10]*o,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromEuler(e){const n=this.elements,i=e.x,r=e.y,s=e.z,o=Math.cos(i),a=Math.sin(i),c=Math.cos(r),u=Math.sin(r),p=Math.cos(s),h=Math.sin(s);if(e.order==="XYZ"){const f=o*p,g=o*h,x=a*p,M=a*h;n[0]=c*p,n[4]=-c*h,n[8]=u,n[1]=g+x*u,n[5]=f-M*u,n[9]=-a*c,n[2]=M-f*u,n[6]=x+g*u,n[10]=o*c}else if(e.order==="YXZ"){const f=c*p,g=c*h,x=u*p,M=u*h;n[0]=f+M*a,n[4]=x*a-g,n[8]=o*u,n[1]=o*h,n[5]=o*p,n[9]=-a,n[2]=g*a-x,n[6]=M+f*a,n[10]=o*c}else if(e.order==="ZXY"){const f=c*p,g=c*h,x=u*p,M=u*h;n[0]=f-M*a,n[4]=-o*h,n[8]=x+g*a,n[1]=g+x*a,n[5]=o*p,n[9]=M-f*a,n[2]=-o*u,n[6]=a,n[10]=o*c}else if(e.order==="ZYX"){const f=o*p,g=o*h,x=a*p,M=a*h;n[0]=c*p,n[4]=x*u-g,n[8]=f*u+M,n[1]=c*h,n[5]=M*u+f,n[9]=g*u-x,n[2]=-u,n[6]=a*c,n[10]=o*c}else if(e.order==="YZX"){const f=o*c,g=o*u,x=a*c,M=a*u;n[0]=c*p,n[4]=M-f*h,n[8]=x*h+g,n[1]=h,n[5]=o*p,n[9]=-a*p,n[2]=-u*p,n[6]=g*h+x,n[10]=f-M*h}else if(e.order==="XZY"){const f=o*c,g=o*u,x=a*c,M=a*u;n[0]=c*p,n[4]=-h,n[8]=u*p,n[1]=f*h+M,n[5]=o*p,n[9]=g*h-x,n[2]=x*h-g,n[6]=a*p,n[10]=M*h+f}return n[3]=0,n[7]=0,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromQuaternion(e){return this.compose(sS,e,oS)}lookAt(e,n,i){const r=this.elements;return _n.subVectors(e,n),_n.lengthSq()===0&&(_n.z=1),_n.normalize(),ji.crossVectors(i,_n),ji.lengthSq()===0&&(Math.abs(i.z)===1?_n.x+=1e-4:_n.z+=1e-4,_n.normalize(),ji.crossVectors(i,_n)),ji.normalize(),Oa.crossVectors(_n,ji),r[0]=ji.x,r[4]=Oa.x,r[8]=_n.x,r[1]=ji.y,r[5]=Oa.y,r[9]=_n.y,r[2]=ji.z,r[6]=Oa.z,r[10]=_n.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){const i=e.elements,r=n.elements,s=this.elements,o=i[0],a=i[4],c=i[8],u=i[12],p=i[1],h=i[5],f=i[9],g=i[13],x=i[2],M=i[6],v=i[10],d=i[14],m=i[3],_=i[7],b=i[11],w=i[15],A=r[0],E=r[4],y=r[8],C=r[12],P=r[1],I=r[5],F=r[9],B=r[13],W=r[2],V=r[6],G=r[10],z=r[14],j=r[3],$=r[7],Q=r[11],se=r[15];return s[0]=o*A+a*P+c*W+u*j,s[4]=o*E+a*I+c*V+u*$,s[8]=o*y+a*F+c*G+u*Q,s[12]=o*C+a*B+c*z+u*se,s[1]=p*A+h*P+f*W+g*j,s[5]=p*E+h*I+f*V+g*$,s[9]=p*y+h*F+f*G+g*Q,s[13]=p*C+h*B+f*z+g*se,s[2]=x*A+M*P+v*W+d*j,s[6]=x*E+M*I+v*V+d*$,s[10]=x*y+M*F+v*G+d*Q,s[14]=x*C+M*B+v*z+d*se,s[3]=m*A+_*P+b*W+w*j,s[7]=m*E+_*I+b*V+w*$,s[11]=m*y+_*F+b*G+w*Q,s[15]=m*C+_*B+b*z+w*se,this}multiplyScalar(e){const n=this.elements;return n[0]*=e,n[4]*=e,n[8]*=e,n[12]*=e,n[1]*=e,n[5]*=e,n[9]*=e,n[13]*=e,n[2]*=e,n[6]*=e,n[10]*=e,n[14]*=e,n[3]*=e,n[7]*=e,n[11]*=e,n[15]*=e,this}determinant(){const e=this.elements,n=e[0],i=e[4],r=e[8],s=e[12],o=e[1],a=e[5],c=e[9],u=e[13],p=e[2],h=e[6],f=e[10],g=e[14],x=e[3],M=e[7],v=e[11],d=e[15],m=c*g-u*f,_=a*g-u*h,b=a*f-c*h,w=o*g-u*p,A=o*f-c*p,E=o*h-a*p;return n*(M*m-v*_+d*b)-i*(x*m-v*w+d*A)+r*(x*_-M*w+d*E)-s*(x*b-M*A+v*E)}transpose(){const e=this.elements;let n;return n=e[1],e[1]=e[4],e[4]=n,n=e[2],e[2]=e[8],e[8]=n,n=e[6],e[6]=e[9],e[9]=n,n=e[3],e[3]=e[12],e[12]=n,n=e[7],e[7]=e[13],e[13]=n,n=e[11],e[11]=e[14],e[14]=n,this}setPosition(e,n,i){const r=this.elements;return e.isVector3?(r[12]=e.x,r[13]=e.y,r[14]=e.z):(r[12]=e,r[13]=n,r[14]=i),this}invert(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],c=e[6],u=e[7],p=e[8],h=e[9],f=e[10],g=e[11],x=e[12],M=e[13],v=e[14],d=e[15],m=n*a-i*o,_=n*c-r*o,b=n*u-s*o,w=i*c-r*a,A=i*u-s*a,E=r*u-s*c,y=p*M-h*x,C=p*v-f*x,P=p*d-g*x,I=h*v-f*M,F=h*d-g*M,B=f*d-g*v,W=m*B-_*F+b*I+w*P-A*C+E*y;if(W===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const V=1/W;return e[0]=(a*B-c*F+u*I)*V,e[1]=(r*F-i*B-s*I)*V,e[2]=(M*E-v*A+d*w)*V,e[3]=(f*A-h*E-g*w)*V,e[4]=(c*P-o*B-u*C)*V,e[5]=(n*B-r*P+s*C)*V,e[6]=(v*b-x*E-d*_)*V,e[7]=(p*E-f*b+g*_)*V,e[8]=(o*F-a*P+u*y)*V,e[9]=(i*P-n*F-s*y)*V,e[10]=(x*A-M*b+d*m)*V,e[11]=(h*b-p*A-g*m)*V,e[12]=(a*C-o*I-c*y)*V,e[13]=(n*I-i*C+r*y)*V,e[14]=(M*_-x*w-v*m)*V,e[15]=(p*w-h*_+f*m)*V,this}scale(e){const n=this.elements,i=e.x,r=e.y,s=e.z;return n[0]*=i,n[4]*=r,n[8]*=s,n[1]*=i,n[5]*=r,n[9]*=s,n[2]*=i,n[6]*=r,n[10]*=s,n[3]*=i,n[7]*=r,n[11]*=s,this}getMaxScaleOnAxis(){const e=this.elements,n=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],r=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(n,i,r))}makeTranslation(e,n,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,n,0,0,1,i,0,0,0,1),this}makeRotationX(e){const n=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,n,-i,0,0,i,n,0,0,0,0,1),this}makeRotationY(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,0,i,0,0,1,0,0,-i,0,n,0,0,0,0,1),this}makeRotationZ(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,0,i,n,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,n){const i=Math.cos(n),r=Math.sin(n),s=1-i,o=e.x,a=e.y,c=e.z,u=s*o,p=s*a;return this.set(u*o+i,u*a-r*c,u*c+r*a,0,u*a+r*c,p*a+i,p*c-r*o,0,u*c-r*a,p*c+r*o,s*c*c+i,0,0,0,0,1),this}makeScale(e,n,i){return this.set(e,0,0,0,0,n,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,n,i,r,s,o){return this.set(1,i,s,0,e,1,o,0,n,r,1,0,0,0,0,1),this}compose(e,n,i){const r=this.elements,s=n._x,o=n._y,a=n._z,c=n._w,u=s+s,p=o+o,h=a+a,f=s*u,g=s*p,x=s*h,M=o*p,v=o*h,d=a*h,m=c*u,_=c*p,b=c*h,w=i.x,A=i.y,E=i.z;return r[0]=(1-(M+d))*w,r[1]=(g+b)*w,r[2]=(x-_)*w,r[3]=0,r[4]=(g-b)*A,r[5]=(1-(f+d))*A,r[6]=(v+m)*A,r[7]=0,r[8]=(x+_)*E,r[9]=(v-m)*E,r[10]=(1-(f+M))*E,r[11]=0,r[12]=e.x,r[13]=e.y,r[14]=e.z,r[15]=1,this}decompose(e,n,i){const r=this.elements;e.x=r[12],e.y=r[13],e.z=r[14];const s=this.determinant();if(s===0)return i.set(1,1,1),n.identity(),this;let o=es.set(r[0],r[1],r[2]).length();const a=es.set(r[4],r[5],r[6]).length(),c=es.set(r[8],r[9],r[10]).length();s<0&&(o=-o),Bn.copy(this);const u=1/o,p=1/a,h=1/c;return Bn.elements[0]*=u,Bn.elements[1]*=u,Bn.elements[2]*=u,Bn.elements[4]*=p,Bn.elements[5]*=p,Bn.elements[6]*=p,Bn.elements[8]*=h,Bn.elements[9]*=h,Bn.elements[10]*=h,n.setFromRotationMatrix(Bn),i.x=o,i.y=a,i.z=c,this}makePerspective(e,n,i,r,s,o,a=ai,c=!1){const u=this.elements,p=2*s/(n-e),h=2*s/(i-r),f=(n+e)/(n-e),g=(i+r)/(i-r);let x,M;if(c)x=s/(o-s),M=o*s/(o-s);else if(a===ai)x=-(o+s)/(o-s),M=-2*o*s/(o-s);else if(a===na)x=-o/(o-s),M=-o*s/(o-s);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return u[0]=p,u[4]=0,u[8]=f,u[12]=0,u[1]=0,u[5]=h,u[9]=g,u[13]=0,u[2]=0,u[6]=0,u[10]=x,u[14]=M,u[3]=0,u[7]=0,u[11]=-1,u[15]=0,this}makeOrthographic(e,n,i,r,s,o,a=ai,c=!1){const u=this.elements,p=2/(n-e),h=2/(i-r),f=-(n+e)/(n-e),g=-(i+r)/(i-r);let x,M;if(c)x=1/(o-s),M=o/(o-s);else if(a===ai)x=-2/(o-s),M=-(o+s)/(o-s);else if(a===na)x=-1/(o-s),M=-s/(o-s);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return u[0]=p,u[4]=0,u[8]=0,u[12]=f,u[1]=0,u[5]=h,u[9]=0,u[13]=g,u[2]=0,u[6]=0,u[10]=x,u[14]=M,u[3]=0,u[7]=0,u[11]=0,u[15]=1,this}equals(e){const n=this.elements,i=e.elements;for(let r=0;r<16;r++)if(n[r]!==i[r])return!1;return!0}fromArray(e,n=0){for(let i=0;i<16;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){const i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e[n+9]=i[9],e[n+10]=i[10],e[n+11]=i[11],e[n+12]=i[12],e[n+13]=i[13],e[n+14]=i[14],e[n+15]=i[15],e}}const es=new H,Bn=new _t,sS=new H(0,0,0),oS=new H(1,1,1),ji=new H,Oa=new H,_n=new H,Am=new _t,Rm=new to;class pi{constructor(e=0,n=0,i=0,r=pi.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=n,this._z=i,this._order=r}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,n,i,r=this._order){return this._x=e,this._y=n,this._z=i,this._order=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,n=this._order,i=!0){const r=e.elements,s=r[0],o=r[4],a=r[8],c=r[1],u=r[5],p=r[9],h=r[2],f=r[6],g=r[10];switch(n){case"XYZ":this._y=Math.asin(nt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-p,g),this._z=Math.atan2(-o,s)):(this._x=Math.atan2(f,u),this._z=0);break;case"YXZ":this._x=Math.asin(-nt(p,-1,1)),Math.abs(p)<.9999999?(this._y=Math.atan2(a,g),this._z=Math.atan2(c,u)):(this._y=Math.atan2(-h,s),this._z=0);break;case"ZXY":this._x=Math.asin(nt(f,-1,1)),Math.abs(f)<.9999999?(this._y=Math.atan2(-h,g),this._z=Math.atan2(-o,u)):(this._y=0,this._z=Math.atan2(c,s));break;case"ZYX":this._y=Math.asin(-nt(h,-1,1)),Math.abs(h)<.9999999?(this._x=Math.atan2(f,g),this._z=Math.atan2(c,s)):(this._x=0,this._z=Math.atan2(-o,u));break;case"YZX":this._z=Math.asin(nt(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-p,u),this._y=Math.atan2(-h,s)):(this._x=0,this._y=Math.atan2(a,g));break;case"XZY":this._z=Math.asin(-nt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(f,u),this._y=Math.atan2(a,s)):(this._x=Math.atan2(-p,g),this._y=0);break;default:He("Euler: .setFromRotationMatrix() encountered an unknown order: "+n)}return this._order=n,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,n,i){return Am.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Am,n,i)}setFromVector3(e,n=this._order){return this.set(e.x,e.y,e.z,n)}reorder(e){return Rm.setFromEuler(this),this.setFromQuaternion(Rm,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}pi.DEFAULT_ORDER="XYZ";class nx{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let aS=0;const Im=new H,ts=new to,vi=new _t,za=new H,fo=new H,lS=new H,cS=new to,Pm=new H(1,0,0),Dm=new H(0,1,0),Lm=new H(0,0,1),Fm={type:"added"},uS={type:"removed"},ns={type:"childadded",child:null},yu={type:"childremoved",child:null};class Gt extends eo{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:aS++}),this.uuid=cr(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Gt.DEFAULT_UP.clone();const e=new H,n=new pi,i=new to,r=new H(1,1,1);function s(){i.setFromEuler(n,!1)}function o(){n.setFromQuaternion(i,void 0,!1)}n._onChange(s),i._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:r},modelViewMatrix:{value:new _t},normalMatrix:{value:new qe}}),this.matrix=new _t,this.matrixWorld=new _t,this.matrixAutoUpdate=Gt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Gt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new nx,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,n){this.quaternion.setFromAxisAngle(e,n)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,n){return ts.setFromAxisAngle(e,n),this.quaternion.multiply(ts),this}rotateOnWorldAxis(e,n){return ts.setFromAxisAngle(e,n),this.quaternion.premultiply(ts),this}rotateX(e){return this.rotateOnAxis(Pm,e)}rotateY(e){return this.rotateOnAxis(Dm,e)}rotateZ(e){return this.rotateOnAxis(Lm,e)}translateOnAxis(e,n){return Im.copy(e).applyQuaternion(this.quaternion),this.position.add(Im.multiplyScalar(n)),this}translateX(e){return this.translateOnAxis(Pm,e)}translateY(e){return this.translateOnAxis(Dm,e)}translateZ(e){return this.translateOnAxis(Lm,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(vi.copy(this.matrixWorld).invert())}lookAt(e,n,i){e.isVector3?za.copy(e):za.set(e,n,i);const r=this.parent;this.updateWorldMatrix(!0,!1),fo.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?vi.lookAt(fo,za,this.up):vi.lookAt(za,fo,this.up),this.quaternion.setFromRotationMatrix(vi),r&&(vi.extractRotation(r.matrixWorld),ts.setFromRotationMatrix(vi),this.quaternion.premultiply(ts.invert()))}add(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.add(arguments[n]);return this}return e===this?(it("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Fm),ns.child=e,this.dispatchEvent(ns),ns.child=null):it("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}const n=this.children.indexOf(e);return n!==-1&&(e.parent=null,this.children.splice(n,1),e.dispatchEvent(uS),yu.child=e,this.dispatchEvent(yu),yu.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),vi.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),vi.multiply(e.parent.matrixWorld)),e.applyMatrix4(vi),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Fm),ns.child=e,this.dispatchEvent(ns),ns.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,n){if(this[e]===n)return this;for(let i=0,r=this.children.length;i<r;i++){const o=this.children[i].getObjectByProperty(e,n);if(o!==void 0)return o}}getObjectsByProperty(e,n,i=[]){this[e]===n&&i.push(this);const r=this.children;for(let s=0,o=r.length;s<o;s++)r[s].getObjectsByProperty(e,n,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(fo,e,lS),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(fo,cS,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const n=this.matrixWorld.elements;return e.set(n[8],n[9],n[10]).normalize()}raycast(){}traverse(e){e(this);const n=this.children;for(let i=0,r=n.length;i<r;i++)n[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const n=this.children;for(let i=0,r=n.length;i<r;i++)n[i].traverseVisible(e)}traverseAncestors(e){const n=this.parent;n!==null&&(e(n),n.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);const e=this.pivot;if(e!==null){const n=e.x,i=e.y,r=e.z,s=this.matrix.elements;s[12]+=n-s[0]*n-s[4]*i-s[8]*r,s[13]+=i-s[1]*n-s[5]*i-s[9]*r,s[14]+=r-s[2]*n-s[6]*i-s[10]*r}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const n=this.children;for(let i=0,r=n.length;i<r;i++)n[i].updateMatrixWorld(e)}updateWorldMatrix(e,n){const i=this.parent;if(e===!0&&i!==null&&i.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),n===!0){const r=this.children;for(let s=0,o=r.length;s<o;s++)r[s].updateWorldMatrix(!1,!0)}}toJSON(e){const n=e===void 0||typeof e=="string",i={};n&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const r={};r.uuid=this.uuid,r.type=this.type,this.name!==""&&(r.name=this.name),this.castShadow===!0&&(r.castShadow=!0),this.receiveShadow===!0&&(r.receiveShadow=!0),this.visible===!1&&(r.visible=!1),this.frustumCulled===!1&&(r.frustumCulled=!1),this.renderOrder!==0&&(r.renderOrder=this.renderOrder),this.static!==!1&&(r.static=this.static),Object.keys(this.userData).length>0&&(r.userData=this.userData),r.layers=this.layers.mask,r.matrix=this.matrix.toArray(),r.up=this.up.toArray(),this.pivot!==null&&(r.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(r.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(r.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(r.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(r.type="InstancedMesh",r.count=this.count,r.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(r.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(r.type="BatchedMesh",r.perObjectFrustumCulled=this.perObjectFrustumCulled,r.sortObjects=this.sortObjects,r.drawRanges=this._drawRanges,r.reservedRanges=this._reservedRanges,r.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),r.instanceInfo=this._instanceInfo.map(a=>({...a})),r.availableInstanceIds=this._availableInstanceIds.slice(),r.availableGeometryIds=this._availableGeometryIds.slice(),r.nextIndexStart=this._nextIndexStart,r.nextVertexStart=this._nextVertexStart,r.geometryCount=this._geometryCount,r.maxInstanceCount=this._maxInstanceCount,r.maxVertexCount=this._maxVertexCount,r.maxIndexCount=this._maxIndexCount,r.geometryInitialized=this._geometryInitialized,r.matricesTexture=this._matricesTexture.toJSON(e),r.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(r.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(r.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(r.boundingBox=this.boundingBox.toJSON()));function s(a,c){return a[c.uuid]===void 0&&(a[c.uuid]=c.toJSON(e)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?r.background=this.background.toJSON():this.background.isTexture&&(r.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(r.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){r.geometry=s(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const c=a.shapes;if(Array.isArray(c))for(let u=0,p=c.length;u<p;u++){const h=c[u];s(e.shapes,h)}else s(e.shapes,c)}}if(this.isSkinnedMesh&&(r.bindMode=this.bindMode,r.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(s(e.skeletons,this.skeleton),r.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let c=0,u=this.material.length;c<u;c++)a.push(s(e.materials,this.material[c]));r.material=a}else r.material=s(e.materials,this.material);if(this.children.length>0){r.children=[];for(let a=0;a<this.children.length;a++)r.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){r.animations=[];for(let a=0;a<this.animations.length;a++){const c=this.animations[a];r.animations.push(s(e.animations,c))}}if(n){const a=o(e.geometries),c=o(e.materials),u=o(e.textures),p=o(e.images),h=o(e.shapes),f=o(e.skeletons),g=o(e.animations),x=o(e.nodes);a.length>0&&(i.geometries=a),c.length>0&&(i.materials=c),u.length>0&&(i.textures=u),p.length>0&&(i.images=p),h.length>0&&(i.shapes=h),f.length>0&&(i.skeletons=f),g.length>0&&(i.animations=g),x.length>0&&(i.nodes=x)}return i.object=r,i;function o(a){const c=[];for(const u in a){const p=a[u];delete p.metadata,c.push(p)}return c}}clone(e){return new this.constructor().copy(this,e)}copy(e,n=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),e.pivot!==null&&(this.pivot=e.pivot.clone()),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),n===!0)for(let i=0;i<e.children.length;i++){const r=e.children[i];this.add(r.clone())}return this}}Gt.DEFAULT_UP=new H(0,1,0);Gt.DEFAULT_MATRIX_AUTO_UPDATE=!0;Gt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;class Ba extends Gt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const dS={type:"move"};class Su{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Ba,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Ba,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new H,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new H),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Ba,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new H,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new H),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const n=this._hand;if(n)for(const i of e.hand.values())this._getHandJoint(n,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,n,i){let r=null,s=null,o=null;const a=this._targetRay,c=this._grip,u=this._hand;if(e&&n.session.visibilityState!=="visible-blurred"){if(u&&e.hand){o=!0;for(const M of e.hand.values()){const v=n.getJointPose(M,i),d=this._getHandJoint(u,M);v!==null&&(d.matrix.fromArray(v.transform.matrix),d.matrix.decompose(d.position,d.rotation,d.scale),d.matrixWorldNeedsUpdate=!0,d.jointRadius=v.radius),d.visible=v!==null}const p=u.joints["index-finger-tip"],h=u.joints["thumb-tip"],f=p.position.distanceTo(h.position),g=.02,x=.005;u.inputState.pinching&&f>g+x?(u.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!u.inputState.pinching&&f<=g-x&&(u.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else c!==null&&e.gripSpace&&(s=n.getPose(e.gripSpace,i),s!==null&&(c.matrix.fromArray(s.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,s.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(s.linearVelocity)):c.hasLinearVelocity=!1,s.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(s.angularVelocity)):c.hasAngularVelocity=!1));a!==null&&(r=n.getPose(e.targetRaySpace,i),r===null&&s!==null&&(r=s),r!==null&&(a.matrix.fromArray(r.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,r.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(r.linearVelocity)):a.hasLinearVelocity=!1,r.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(r.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(dS)))}return a!==null&&(a.visible=r!==null),c!==null&&(c.visible=s!==null),u!==null&&(u.visible=o!==null),this}_getHandJoint(e,n){if(e.joints[n.jointName]===void 0){const i=new Ba;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[n.jointName]=i,e.add(i)}return e.joints[n.jointName]}}const ix={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Vi={h:0,s:0,l:0},ja={h:0,s:0,l:0};function Mu(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<1/6?t+(e-t)*6*n:n<1/2?e:n<2/3?t+(e-t)*6*(2/3-n):t}class tt{constructor(e,n,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,n,i)}set(e,n,i){if(n===void 0&&i===void 0){const r=e;r&&r.isColor?this.copy(r):typeof r=="number"?this.setHex(r):typeof r=="string"&&this.setStyle(r)}else this.setRGB(e,n,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,n=In){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,rt.colorSpaceToWorking(this,n),this}setRGB(e,n,i,r=rt.workingColorSpace){return this.r=e,this.g=n,this.b=i,rt.colorSpaceToWorking(this,r),this}setHSL(e,n,i,r=rt.workingColorSpace){if(e=Jy(e,1),n=nt(n,0,1),i=nt(i,0,1),n===0)this.r=this.g=this.b=i;else{const s=i<=.5?i*(1+n):i+n-i*n,o=2*i-s;this.r=Mu(o,s,e+1/3),this.g=Mu(o,s,e),this.b=Mu(o,s,e-1/3)}return rt.colorSpaceToWorking(this,r),this}setStyle(e,n=In){function i(s){s!==void 0&&parseFloat(s)<1&&He("Color: Alpha component of "+e+" will be ignored.")}let r;if(r=/^(\w+)\(([^\)]*)\)/.exec(e)){let s;const o=r[1],a=r[2];switch(o){case"rgb":case"rgba":if(s=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(s[4]),this.setRGB(Math.min(255,parseInt(s[1],10))/255,Math.min(255,parseInt(s[2],10))/255,Math.min(255,parseInt(s[3],10))/255,n);if(s=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(s[4]),this.setRGB(Math.min(100,parseInt(s[1],10))/100,Math.min(100,parseInt(s[2],10))/100,Math.min(100,parseInt(s[3],10))/100,n);break;case"hsl":case"hsla":if(s=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(s[4]),this.setHSL(parseFloat(s[1])/360,parseFloat(s[2])/100,parseFloat(s[3])/100,n);break;default:He("Color: Unknown color model "+e)}}else if(r=/^\#([A-Fa-f\d]+)$/.exec(e)){const s=r[1],o=s.length;if(o===3)return this.setRGB(parseInt(s.charAt(0),16)/15,parseInt(s.charAt(1),16)/15,parseInt(s.charAt(2),16)/15,n);if(o===6)return this.setHex(parseInt(s,16),n);He("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,n);return this}setColorName(e,n=In){const i=ix[e.toLowerCase()];return i!==void 0?this.setHex(i,n):He("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Ii(e.r),this.g=Ii(e.g),this.b=Ii(e.b),this}copyLinearToSRGB(e){return this.r=ks(e.r),this.g=ks(e.g),this.b=ks(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=In){return rt.workingToColorSpace(Zt.copy(this),e),Math.round(nt(Zt.r*255,0,255))*65536+Math.round(nt(Zt.g*255,0,255))*256+Math.round(nt(Zt.b*255,0,255))}getHexString(e=In){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,n=rt.workingColorSpace){rt.workingToColorSpace(Zt.copy(this),n);const i=Zt.r,r=Zt.g,s=Zt.b,o=Math.max(i,r,s),a=Math.min(i,r,s);let c,u;const p=(a+o)/2;if(a===o)c=0,u=0;else{const h=o-a;switch(u=p<=.5?h/(o+a):h/(2-o-a),o){case i:c=(r-s)/h+(r<s?6:0);break;case r:c=(s-i)/h+2;break;case s:c=(i-r)/h+4;break}c/=6}return e.h=c,e.s=u,e.l=p,e}getRGB(e,n=rt.workingColorSpace){return rt.workingToColorSpace(Zt.copy(this),n),e.r=Zt.r,e.g=Zt.g,e.b=Zt.b,e}getStyle(e=In){rt.workingToColorSpace(Zt.copy(this),e);const n=Zt.r,i=Zt.g,r=Zt.b;return e!==In?`color(${e} ${n.toFixed(3)} ${i.toFixed(3)} ${r.toFixed(3)})`:`rgb(${Math.round(n*255)},${Math.round(i*255)},${Math.round(r*255)})`}offsetHSL(e,n,i){return this.getHSL(Vi),this.setHSL(Vi.h+e,Vi.s+n,Vi.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,n){return this.r=e.r+n.r,this.g=e.g+n.g,this.b=e.b+n.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,n){return this.r+=(e.r-this.r)*n,this.g+=(e.g-this.g)*n,this.b+=(e.b-this.b)*n,this}lerpColors(e,n,i){return this.r=e.r+(n.r-e.r)*i,this.g=e.g+(n.g-e.g)*i,this.b=e.b+(n.b-e.b)*i,this}lerpHSL(e,n){this.getHSL(Vi),e.getHSL(ja);const i=mu(Vi.h,ja.h,n),r=mu(Vi.s,ja.s,n),s=mu(Vi.l,ja.l,n);return this.setHSL(i,r,s),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const n=this.r,i=this.g,r=this.b,s=e.elements;return this.r=s[0]*n+s[3]*i+s[6]*r,this.g=s[1]*n+s[4]*i+s[7]*r,this.b=s[2]*n+s[5]*i+s[8]*r,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,n=0){return this.r=e[n],this.g=e[n+1],this.b=e[n+2],this}toArray(e=[],n=0){return e[n]=this.r,e[n+1]=this.g,e[n+2]=this.b,e}fromBufferAttribute(e,n){return this.r=e.getX(n),this.g=e.getY(n),this.b=e.getZ(n),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Zt=new tt;tt.NAMES=ix;class fS extends Gt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new pi,this.environmentIntensity=1,this.environmentRotation=new pi,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,n){return super.copy(e,n),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const n=super.toJSON(e);return this.fog!==null&&(n.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(n.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(n.object.backgroundIntensity=this.backgroundIntensity),n.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(n.object.environmentIntensity=this.environmentIntensity),n.object.environmentRotation=this.environmentRotation.toArray(),n}}const jn=new H,_i=new H,bu=new H,yi=new H,is=new H,rs=new H,Nm=new H,Eu=new H,Tu=new H,wu=new H,Cu=new wt,Au=new wt,Ru=new wt;class Ln{constructor(e=new H,n=new H,i=new H){this.a=e,this.b=n,this.c=i}static getNormal(e,n,i,r){r.subVectors(i,n),jn.subVectors(e,n),r.cross(jn);const s=r.lengthSq();return s>0?r.multiplyScalar(1/Math.sqrt(s)):r.set(0,0,0)}static getBarycoord(e,n,i,r,s){jn.subVectors(r,n),_i.subVectors(i,n),bu.subVectors(e,n);const o=jn.dot(jn),a=jn.dot(_i),c=jn.dot(bu),u=_i.dot(_i),p=_i.dot(bu),h=o*u-a*a;if(h===0)return s.set(0,0,0),null;const f=1/h,g=(u*c-a*p)*f,x=(o*p-a*c)*f;return s.set(1-g-x,x,g)}static containsPoint(e,n,i,r){return this.getBarycoord(e,n,i,r,yi)===null?!1:yi.x>=0&&yi.y>=0&&yi.x+yi.y<=1}static getInterpolation(e,n,i,r,s,o,a,c){return this.getBarycoord(e,n,i,r,yi)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(s,yi.x),c.addScaledVector(o,yi.y),c.addScaledVector(a,yi.z),c)}static getInterpolatedAttribute(e,n,i,r,s,o){return Cu.setScalar(0),Au.setScalar(0),Ru.setScalar(0),Cu.fromBufferAttribute(e,n),Au.fromBufferAttribute(e,i),Ru.fromBufferAttribute(e,r),o.setScalar(0),o.addScaledVector(Cu,s.x),o.addScaledVector(Au,s.y),o.addScaledVector(Ru,s.z),o}static isFrontFacing(e,n,i,r){return jn.subVectors(i,n),_i.subVectors(e,n),jn.cross(_i).dot(r)<0}set(e,n,i){return this.a.copy(e),this.b.copy(n),this.c.copy(i),this}setFromPointsAndIndices(e,n,i,r){return this.a.copy(e[n]),this.b.copy(e[i]),this.c.copy(e[r]),this}setFromAttributeAndIndices(e,n,i,r){return this.a.fromBufferAttribute(e,n),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,r),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return jn.subVectors(this.c,this.b),_i.subVectors(this.a,this.b),jn.cross(_i).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return Ln.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,n){return Ln.getBarycoord(e,this.a,this.b,this.c,n)}getInterpolation(e,n,i,r,s){return Ln.getInterpolation(e,this.a,this.b,this.c,n,i,r,s)}containsPoint(e){return Ln.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return Ln.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,n){const i=this.a,r=this.b,s=this.c;let o,a;is.subVectors(r,i),rs.subVectors(s,i),Eu.subVectors(e,i);const c=is.dot(Eu),u=rs.dot(Eu);if(c<=0&&u<=0)return n.copy(i);Tu.subVectors(e,r);const p=is.dot(Tu),h=rs.dot(Tu);if(p>=0&&h<=p)return n.copy(r);const f=c*h-p*u;if(f<=0&&c>=0&&p<=0)return o=c/(c-p),n.copy(i).addScaledVector(is,o);wu.subVectors(e,s);const g=is.dot(wu),x=rs.dot(wu);if(x>=0&&g<=x)return n.copy(s);const M=g*u-c*x;if(M<=0&&u>=0&&x<=0)return a=u/(u-x),n.copy(i).addScaledVector(rs,a);const v=p*x-g*h;if(v<=0&&h-p>=0&&g-x>=0)return Nm.subVectors(s,r),a=(h-p)/(h-p+(g-x)),n.copy(r).addScaledVector(Nm,a);const d=1/(v+M+f);return o=M*d,a=f*d,n.copy(i).addScaledVector(is,o).addScaledVector(rs,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}class fa{constructor(e=new H(1/0,1/0,1/0),n=new H(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=n}set(e,n){return this.min.copy(e),this.max.copy(n),this}setFromArray(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n+=3)this.expandByPoint(Vn.fromArray(e,n));return this}setFromBufferAttribute(e){this.makeEmpty();for(let n=0,i=e.count;n<i;n++)this.expandByPoint(Vn.fromBufferAttribute(e,n));return this}setFromPoints(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n++)this.expandByPoint(e[n]);return this}setFromCenterAndSize(e,n){const i=Vn.copy(n).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,n=!1){return this.makeEmpty(),this.expandByObject(e,n)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,n=!1){e.updateWorldMatrix(!1,!1);const i=e.geometry;if(i!==void 0){const s=i.getAttribute("position");if(n===!0&&s!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=s.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,Vn):Vn.fromBufferAttribute(s,o),Vn.applyMatrix4(e.matrixWorld),this.expandByPoint(Vn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Va.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Va.copy(i.boundingBox)),Va.applyMatrix4(e.matrixWorld),this.union(Va)}const r=e.children;for(let s=0,o=r.length;s<o;s++)this.expandByObject(r[s],n);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,n){return n.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Vn),Vn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let n,i;return e.normal.x>0?(n=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(n=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(n+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(n+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(n+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(n+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),n<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(po),Ha.subVectors(this.max,po),ss.subVectors(e.a,po),os.subVectors(e.b,po),as.subVectors(e.c,po),Hi.subVectors(os,ss),Gi.subVectors(as,os),Sr.subVectors(ss,as);let n=[0,-Hi.z,Hi.y,0,-Gi.z,Gi.y,0,-Sr.z,Sr.y,Hi.z,0,-Hi.x,Gi.z,0,-Gi.x,Sr.z,0,-Sr.x,-Hi.y,Hi.x,0,-Gi.y,Gi.x,0,-Sr.y,Sr.x,0];return!Iu(n,ss,os,as,Ha)||(n=[1,0,0,0,1,0,0,0,1],!Iu(n,ss,os,as,Ha))?!1:(Ga.crossVectors(Hi,Gi),n=[Ga.x,Ga.y,Ga.z],Iu(n,ss,os,as,Ha))}clampPoint(e,n){return n.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Vn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Vn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Si[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Si[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Si[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Si[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Si[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Si[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Si[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Si[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Si),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Si=[new H,new H,new H,new H,new H,new H,new H,new H],Vn=new H,Va=new fa,ss=new H,os=new H,as=new H,Hi=new H,Gi=new H,Sr=new H,po=new H,Ha=new H,Ga=new H,Mr=new H;function Iu(t,e,n,i,r){for(let s=0,o=t.length-3;s<=o;s+=3){Mr.fromArray(t,s);const a=r.x*Math.abs(Mr.x)+r.y*Math.abs(Mr.y)+r.z*Math.abs(Mr.z),c=e.dot(Mr),u=n.dot(Mr),p=i.dot(Mr);if(Math.max(-Math.max(c,u,p),Math.min(c,u,p))>a)return!1}return!0}const Rt=new H,Wa=new Ze;let pS=0;class Kn{constructor(e,n,i=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:pS++}),this.name="",this.array=e,this.itemSize=n,this.count=e!==void 0?e.length/n:0,this.normalized=i,this.usage=Pf,this.updateRanges=[],this.gpuType=oi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,n,i){e*=this.itemSize,i*=n.itemSize;for(let r=0,s=this.itemSize;r<s;r++)this.array[e+r]=n.array[i+r];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let n=0,i=this.count;n<i;n++)Wa.fromBufferAttribute(this,n),Wa.applyMatrix3(e),this.setXY(n,Wa.x,Wa.y);else if(this.itemSize===3)for(let n=0,i=this.count;n<i;n++)Rt.fromBufferAttribute(this,n),Rt.applyMatrix3(e),this.setXYZ(n,Rt.x,Rt.y,Rt.z);return this}applyMatrix4(e){for(let n=0,i=this.count;n<i;n++)Rt.fromBufferAttribute(this,n),Rt.applyMatrix4(e),this.setXYZ(n,Rt.x,Rt.y,Rt.z);return this}applyNormalMatrix(e){for(let n=0,i=this.count;n<i;n++)Rt.fromBufferAttribute(this,n),Rt.applyNormalMatrix(e),this.setXYZ(n,Rt.x,Rt.y,Rt.z);return this}transformDirection(e){for(let n=0,i=this.count;n<i;n++)Rt.fromBufferAttribute(this,n),Rt.transformDirection(e),this.setXYZ(n,Rt.x,Rt.y,Rt.z);return this}set(e,n=0){return this.array.set(e,n),this}getComponent(e,n){let i=this.array[e*this.itemSize+n];return this.normalized&&(i=si(i,this.array)),i}setComponent(e,n,i){return this.normalized&&(i=ft(i,this.array)),this.array[e*this.itemSize+n]=i,this}getX(e){let n=this.array[e*this.itemSize];return this.normalized&&(n=si(n,this.array)),n}setX(e,n){return this.normalized&&(n=ft(n,this.array)),this.array[e*this.itemSize]=n,this}getY(e){let n=this.array[e*this.itemSize+1];return this.normalized&&(n=si(n,this.array)),n}setY(e,n){return this.normalized&&(n=ft(n,this.array)),this.array[e*this.itemSize+1]=n,this}getZ(e){let n=this.array[e*this.itemSize+2];return this.normalized&&(n=si(n,this.array)),n}setZ(e,n){return this.normalized&&(n=ft(n,this.array)),this.array[e*this.itemSize+2]=n,this}getW(e){let n=this.array[e*this.itemSize+3];return this.normalized&&(n=si(n,this.array)),n}setW(e,n){return this.normalized&&(n=ft(n,this.array)),this.array[e*this.itemSize+3]=n,this}setXY(e,n,i){return e*=this.itemSize,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array)),this.array[e+0]=n,this.array[e+1]=i,this}setXYZ(e,n,i,r){return e*=this.itemSize,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array),r=ft(r,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=r,this}setXYZW(e,n,i,r,s){return e*=this.itemSize,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array),r=ft(r,this.array),s=ft(s,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=r,this.array[e+3]=s,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Pf&&(e.usage=this.usage),e}}class rx extends Kn{constructor(e,n,i){super(new Uint16Array(e),n,i)}}class sx extends Kn{constructor(e,n,i){super(new Uint32Array(e),n,i)}}class Wt extends Kn{constructor(e,n,i){super(new Float32Array(e),n,i)}}const hS=new fa,ho=new H,Pu=new H;class pa{constructor(e=new H,n=-1){this.isSphere=!0,this.center=e,this.radius=n}set(e,n){return this.center.copy(e),this.radius=n,this}setFromPoints(e,n){const i=this.center;n!==void 0?i.copy(n):hS.setFromPoints(e).getCenter(i);let r=0;for(let s=0,o=e.length;s<o;s++)r=Math.max(r,i.distanceToSquared(e[s]));return this.radius=Math.sqrt(r),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const n=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=n*n}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,n){const i=this.center.distanceToSquared(e);return n.copy(e),i>this.radius*this.radius&&(n.sub(this.center).normalize(),n.multiplyScalar(this.radius).add(this.center)),n}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;ho.subVectors(e,this.center);const n=ho.lengthSq();if(n>this.radius*this.radius){const i=Math.sqrt(n),r=(i-this.radius)*.5;this.center.addScaledVector(ho,r/i),this.radius+=r}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Pu.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(ho.copy(e.center).add(Pu)),this.expandByPoint(ho.copy(e.center).sub(Pu))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}let mS=0;const Rn=new _t,Du=new Gt,ls=new H,yn=new fa,mo=new fa,Ut=new H;class rn extends eo{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:mS++}),this.uuid=cr(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new($y(e)?sx:rx)(e,1):this.index=e,this}setIndirect(e,n=0){return this.indirect=e,this.indirectOffset=n,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,n){return this.attributes[e]=n,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,n,i=0){this.groups.push({start:e,count:n,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,n){this.drawRange.start=e,this.drawRange.count=n}applyMatrix4(e){const n=this.attributes.position;n!==void 0&&(n.applyMatrix4(e),n.needsUpdate=!0);const i=this.attributes.normal;if(i!==void 0){const s=new qe().getNormalMatrix(e);i.applyNormalMatrix(s),i.needsUpdate=!0}const r=this.attributes.tangent;return r!==void 0&&(r.transformDirection(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return Rn.makeRotationFromQuaternion(e),this.applyMatrix4(Rn),this}rotateX(e){return Rn.makeRotationX(e),this.applyMatrix4(Rn),this}rotateY(e){return Rn.makeRotationY(e),this.applyMatrix4(Rn),this}rotateZ(e){return Rn.makeRotationZ(e),this.applyMatrix4(Rn),this}translate(e,n,i){return Rn.makeTranslation(e,n,i),this.applyMatrix4(Rn),this}scale(e,n,i){return Rn.makeScale(e,n,i),this.applyMatrix4(Rn),this}lookAt(e){return Du.lookAt(e),Du.updateMatrix(),this.applyMatrix4(Du.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(ls).negate(),this.translate(ls.x,ls.y,ls.z),this}setFromPoints(e){const n=this.getAttribute("position");if(n===void 0){const i=[];for(let r=0,s=e.length;r<s;r++){const o=e[r];i.push(o.x,o.y,o.z||0)}this.setAttribute("position",new Wt(i,3))}else{const i=Math.min(e.length,n.count);for(let r=0;r<i;r++){const s=e[r];n.setXYZ(r,s.x,s.y,s.z||0)}e.length>n.count&&He("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),n.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new fa);const e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){it("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new H(-1/0,-1/0,-1/0),new H(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),n)for(let i=0,r=n.length;i<r;i++){const s=n[i];yn.setFromBufferAttribute(s),this.morphTargetsRelative?(Ut.addVectors(this.boundingBox.min,yn.min),this.boundingBox.expandByPoint(Ut),Ut.addVectors(this.boundingBox.max,yn.max),this.boundingBox.expandByPoint(Ut)):(this.boundingBox.expandByPoint(yn.min),this.boundingBox.expandByPoint(yn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&it('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new pa);const e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){it("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new H,1/0);return}if(e){const i=this.boundingSphere.center;if(yn.setFromBufferAttribute(e),n)for(let s=0,o=n.length;s<o;s++){const a=n[s];mo.setFromBufferAttribute(a),this.morphTargetsRelative?(Ut.addVectors(yn.min,mo.min),yn.expandByPoint(Ut),Ut.addVectors(yn.max,mo.max),yn.expandByPoint(Ut)):(yn.expandByPoint(mo.min),yn.expandByPoint(mo.max))}yn.getCenter(i);let r=0;for(let s=0,o=e.count;s<o;s++)Ut.fromBufferAttribute(e,s),r=Math.max(r,i.distanceToSquared(Ut));if(n)for(let s=0,o=n.length;s<o;s++){const a=n[s],c=this.morphTargetsRelative;for(let u=0,p=a.count;u<p;u++)Ut.fromBufferAttribute(a,u),c&&(ls.fromBufferAttribute(e,u),Ut.add(ls)),r=Math.max(r,i.distanceToSquared(Ut))}this.boundingSphere.radius=Math.sqrt(r),isNaN(this.boundingSphere.radius)&&it('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,n=this.attributes;if(e===null||n.position===void 0||n.normal===void 0||n.uv===void 0){it("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const i=n.position,r=n.normal,s=n.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Kn(new Float32Array(4*i.count),4));const o=this.getAttribute("tangent"),a=[],c=[];for(let y=0;y<i.count;y++)a[y]=new H,c[y]=new H;const u=new H,p=new H,h=new H,f=new Ze,g=new Ze,x=new Ze,M=new H,v=new H;function d(y,C,P){u.fromBufferAttribute(i,y),p.fromBufferAttribute(i,C),h.fromBufferAttribute(i,P),f.fromBufferAttribute(s,y),g.fromBufferAttribute(s,C),x.fromBufferAttribute(s,P),p.sub(u),h.sub(u),g.sub(f),x.sub(f);const I=1/(g.x*x.y-x.x*g.y);isFinite(I)&&(M.copy(p).multiplyScalar(x.y).addScaledVector(h,-g.y).multiplyScalar(I),v.copy(h).multiplyScalar(g.x).addScaledVector(p,-x.x).multiplyScalar(I),a[y].add(M),a[C].add(M),a[P].add(M),c[y].add(v),c[C].add(v),c[P].add(v))}let m=this.groups;m.length===0&&(m=[{start:0,count:e.count}]);for(let y=0,C=m.length;y<C;++y){const P=m[y],I=P.start,F=P.count;for(let B=I,W=I+F;B<W;B+=3)d(e.getX(B+0),e.getX(B+1),e.getX(B+2))}const _=new H,b=new H,w=new H,A=new H;function E(y){w.fromBufferAttribute(r,y),A.copy(w);const C=a[y];_.copy(C),_.sub(w.multiplyScalar(w.dot(C))).normalize(),b.crossVectors(A,C);const I=b.dot(c[y])<0?-1:1;o.setXYZW(y,_.x,_.y,_.z,I)}for(let y=0,C=m.length;y<C;++y){const P=m[y],I=P.start,F=P.count;for(let B=I,W=I+F;B<W;B+=3)E(e.getX(B+0)),E(e.getX(B+1)),E(e.getX(B+2))}}computeVertexNormals(){const e=this.index,n=this.getAttribute("position");if(n!==void 0){let i=this.getAttribute("normal");if(i===void 0)i=new Kn(new Float32Array(n.count*3),3),this.setAttribute("normal",i);else for(let f=0,g=i.count;f<g;f++)i.setXYZ(f,0,0,0);const r=new H,s=new H,o=new H,a=new H,c=new H,u=new H,p=new H,h=new H;if(e)for(let f=0,g=e.count;f<g;f+=3){const x=e.getX(f+0),M=e.getX(f+1),v=e.getX(f+2);r.fromBufferAttribute(n,x),s.fromBufferAttribute(n,M),o.fromBufferAttribute(n,v),p.subVectors(o,s),h.subVectors(r,s),p.cross(h),a.fromBufferAttribute(i,x),c.fromBufferAttribute(i,M),u.fromBufferAttribute(i,v),a.add(p),c.add(p),u.add(p),i.setXYZ(x,a.x,a.y,a.z),i.setXYZ(M,c.x,c.y,c.z),i.setXYZ(v,u.x,u.y,u.z)}else for(let f=0,g=n.count;f<g;f+=3)r.fromBufferAttribute(n,f+0),s.fromBufferAttribute(n,f+1),o.fromBufferAttribute(n,f+2),p.subVectors(o,s),h.subVectors(r,s),p.cross(h),i.setXYZ(f+0,p.x,p.y,p.z),i.setXYZ(f+1,p.x,p.y,p.z),i.setXYZ(f+2,p.x,p.y,p.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let n=0,i=e.count;n<i;n++)Ut.fromBufferAttribute(e,n),Ut.normalize(),e.setXYZ(n,Ut.x,Ut.y,Ut.z)}toNonIndexed(){function e(a,c){const u=a.array,p=a.itemSize,h=a.normalized,f=new u.constructor(c.length*p);let g=0,x=0;for(let M=0,v=c.length;M<v;M++){a.isInterleavedBufferAttribute?g=c[M]*a.data.stride+a.offset:g=c[M]*p;for(let d=0;d<p;d++)f[x++]=u[g++]}return new Kn(f,p,h)}if(this.index===null)return He("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const n=new rn,i=this.index.array,r=this.attributes;for(const a in r){const c=r[a],u=e(c,i);n.setAttribute(a,u)}const s=this.morphAttributes;for(const a in s){const c=[],u=s[a];for(let p=0,h=u.length;p<h;p++){const f=u[p],g=e(f,i);c.push(g)}n.morphAttributes[a]=c}n.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,c=o.length;a<c;a++){const u=o[a];n.addGroup(u.start,u.count,u.materialIndex)}return n}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const c=this.parameters;for(const u in c)c[u]!==void 0&&(e[u]=c[u]);return e}e.data={attributes:{}};const n=this.index;n!==null&&(e.data.index={type:n.array.constructor.name,array:Array.prototype.slice.call(n.array)});const i=this.attributes;for(const c in i){const u=i[c];e.data.attributes[c]=u.toJSON(e.data)}const r={};let s=!1;for(const c in this.morphAttributes){const u=this.morphAttributes[c],p=[];for(let h=0,f=u.length;h<f;h++){const g=u[h];p.push(g.toJSON(e.data))}p.length>0&&(r[c]=p,s=!0)}s&&(e.data.morphAttributes=r,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const n={};this.name=e.name;const i=e.index;i!==null&&this.setIndex(i.clone());const r=e.attributes;for(const u in r){const p=r[u];this.setAttribute(u,p.clone(n))}const s=e.morphAttributes;for(const u in s){const p=[],h=s[u];for(let f=0,g=h.length;f<g;f++)p.push(h[f].clone(n));this.morphAttributes[u]=p}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let u=0,p=o.length;u<p;u++){const h=o[u];this.addGroup(h.start,h.count,h.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const c=e.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}class gS{constructor(e,n){this.isInterleavedBuffer=!0,this.array=e,this.stride=n,this.count=e!==void 0?e.length/n:0,this.usage=Pf,this.updateRanges=[],this.version=0,this.uuid=cr()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,n,i){e*=this.stride,i*=n.stride;for(let r=0,s=this.stride;r<s;r++)this.array[e+r]=n.array[i+r];return this}set(e,n=0){return this.array.set(e,n),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=cr()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);const n=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),i=new this.constructor(n,this.stride);return i.setUsage(this.usage),i}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=cr()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}}const sn=new H;class lc{constructor(e,n,i,r=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=e,this.itemSize=n,this.offset=i,this.normalized=r}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let n=0,i=this.data.count;n<i;n++)sn.fromBufferAttribute(this,n),sn.applyMatrix4(e),this.setXYZ(n,sn.x,sn.y,sn.z);return this}applyNormalMatrix(e){for(let n=0,i=this.count;n<i;n++)sn.fromBufferAttribute(this,n),sn.applyNormalMatrix(e),this.setXYZ(n,sn.x,sn.y,sn.z);return this}transformDirection(e){for(let n=0,i=this.count;n<i;n++)sn.fromBufferAttribute(this,n),sn.transformDirection(e),this.setXYZ(n,sn.x,sn.y,sn.z);return this}getComponent(e,n){let i=this.array[e*this.data.stride+this.offset+n];return this.normalized&&(i=si(i,this.array)),i}setComponent(e,n,i){return this.normalized&&(i=ft(i,this.array)),this.data.array[e*this.data.stride+this.offset+n]=i,this}setX(e,n){return this.normalized&&(n=ft(n,this.array)),this.data.array[e*this.data.stride+this.offset]=n,this}setY(e,n){return this.normalized&&(n=ft(n,this.array)),this.data.array[e*this.data.stride+this.offset+1]=n,this}setZ(e,n){return this.normalized&&(n=ft(n,this.array)),this.data.array[e*this.data.stride+this.offset+2]=n,this}setW(e,n){return this.normalized&&(n=ft(n,this.array)),this.data.array[e*this.data.stride+this.offset+3]=n,this}getX(e){let n=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(n=si(n,this.array)),n}getY(e){let n=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(n=si(n,this.array)),n}getZ(e){let n=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(n=si(n,this.array)),n}getW(e){let n=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(n=si(n,this.array)),n}setXY(e,n,i){return e=e*this.data.stride+this.offset,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array)),this.data.array[e+0]=n,this.data.array[e+1]=i,this}setXYZ(e,n,i,r){return e=e*this.data.stride+this.offset,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array),r=ft(r,this.array)),this.data.array[e+0]=n,this.data.array[e+1]=i,this.data.array[e+2]=r,this}setXYZW(e,n,i,r,s){return e=e*this.data.stride+this.offset,this.normalized&&(n=ft(n,this.array),i=ft(i,this.array),r=ft(r,this.array),s=ft(s,this.array)),this.data.array[e+0]=n,this.data.array[e+1]=i,this.data.array[e+2]=r,this.data.array[e+3]=s,this}clone(e){if(e===void 0){oc("InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");const n=[];for(let i=0;i<this.count;i++){const r=i*this.data.stride+this.offset;for(let s=0;s<this.itemSize;s++)n.push(this.data.array[r+s])}return new Kn(new this.array.constructor(n),this.itemSize,this.normalized)}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.clone(e)),new lc(e.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){oc("InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");const n=[];for(let i=0;i<this.count;i++){const r=i*this.data.stride+this.offset;for(let s=0;s<this.itemSize;s++)n.push(this.data.array[r+s])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:n,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}}let xS=0;class vr extends eo{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:xS++}),this.uuid=cr(),this.name="",this.type="Material",this.blending=Us,this.side=pr,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=jd,this.blendDst=Vd,this.blendEquation=Pr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new tt(0,0,0),this.blendAlpha=0,this.depthFunc=Ws,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Mm,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Jr,this.stencilZFail=Jr,this.stencilZPass=Jr,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const n in e){const i=e[n];if(i===void 0){He(`Material: parameter '${n}' has value of undefined.`);continue}const r=this[n];if(r===void 0){He(`Material: '${n}' is not a property of THREE.${this.type}.`);continue}r&&r.isColor?r.set(i):r&&r.isVector3&&i&&i.isVector3?r.copy(i):this[n]=i}}toJSON(e){const n=e===void 0||typeof e=="string";n&&(e={textures:{},images:{}});const i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==Us&&(i.blending=this.blending),this.side!==pr&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==jd&&(i.blendSrc=this.blendSrc),this.blendDst!==Vd&&(i.blendDst=this.blendDst),this.blendEquation!==Pr&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==Ws&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Mm&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Jr&&(i.stencilFail=this.stencilFail),this.stencilZFail!==Jr&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==Jr&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.allowOverride===!1&&(i.allowOverride=!1),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function r(s){const o=[];for(const a in s){const c=s[a];delete c.metadata,o.push(c)}return o}if(n){const s=r(e.textures),o=r(e.images);s.length>0&&(i.textures=s),o.length>0&&(i.images=o)}return i}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const n=e.clippingPlanes;let i=null;if(n!==null){const r=n.length;i=new Array(r);for(let s=0;s!==r;++s)i[s]=n[s].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class ox extends vr{constructor(e){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new tt(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}let cs;const go=new H,us=new H,ds=new H,fs=new Ze,xo=new Ze,ax=new _t,Xa=new H,vo=new H,qa=new H,Um=new Ze,Lu=new Ze,km=new Ze;class vS extends Gt{constructor(e=new ox){if(super(),this.isSprite=!0,this.type="Sprite",cs===void 0){cs=new rn;const n=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),i=new gS(n,5);cs.setIndex([0,1,2,0,2,3]),cs.setAttribute("position",new lc(i,3,0,!1)),cs.setAttribute("uv",new lc(i,2,3,!1))}this.geometry=cs,this.material=e,this.center=new Ze(.5,.5),this.count=1}raycast(e,n){e.camera===null&&it('Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),us.setFromMatrixScale(this.matrixWorld),ax.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),ds.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&us.multiplyScalar(-ds.z);const i=this.material.rotation;let r,s;i!==0&&(s=Math.cos(i),r=Math.sin(i));const o=this.center;$a(Xa.set(-.5,-.5,0),ds,o,us,r,s),$a(vo.set(.5,-.5,0),ds,o,us,r,s),$a(qa.set(.5,.5,0),ds,o,us,r,s),Um.set(0,0),Lu.set(1,0),km.set(1,1);let a=e.ray.intersectTriangle(Xa,vo,qa,!1,go);if(a===null&&($a(vo.set(-.5,.5,0),ds,o,us,r,s),Lu.set(0,1),a=e.ray.intersectTriangle(Xa,qa,vo,!1,go),a===null))return;const c=e.ray.origin.distanceTo(go);c<e.near||c>e.far||n.push({distance:c,point:go.clone(),uv:Ln.getInterpolation(go,Xa,vo,qa,Um,Lu,km,new Ze),face:null,object:this})}copy(e,n){return super.copy(e,n),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}}function $a(t,e,n,i,r,s){fs.subVectors(t,n).addScalar(.5).multiply(i),r!==void 0?(xo.x=s*fs.x-r*fs.y,xo.y=r*fs.x+s*fs.y):xo.copy(fs),t.copy(e),t.x+=xo.x,t.y+=xo.y,t.applyMatrix4(ax)}const Mi=new H,Fu=new H,Ka=new H,Wi=new H,Nu=new H,Ya=new H,Uu=new H;class Wp{constructor(e=new H,n=new H(0,0,-1)){this.origin=e,this.direction=n}set(e,n){return this.origin.copy(e),this.direction.copy(n),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,n){return n.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Mi)),this}closestPointToPoint(e,n){n.subVectors(e,this.origin);const i=n.dot(this.direction);return i<0?n.copy(this.origin):n.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const n=Mi.subVectors(e,this.origin).dot(this.direction);return n<0?this.origin.distanceToSquared(e):(Mi.copy(this.origin).addScaledVector(this.direction,n),Mi.distanceToSquared(e))}distanceSqToSegment(e,n,i,r){Fu.copy(e).add(n).multiplyScalar(.5),Ka.copy(n).sub(e).normalize(),Wi.copy(this.origin).sub(Fu);const s=e.distanceTo(n)*.5,o=-this.direction.dot(Ka),a=Wi.dot(this.direction),c=-Wi.dot(Ka),u=Wi.lengthSq(),p=Math.abs(1-o*o);let h,f,g,x;if(p>0)if(h=o*c-a,f=o*a-c,x=s*p,h>=0)if(f>=-x)if(f<=x){const M=1/p;h*=M,f*=M,g=h*(h+o*f+2*a)+f*(o*h+f+2*c)+u}else f=s,h=Math.max(0,-(o*f+a)),g=-h*h+f*(f+2*c)+u;else f=-s,h=Math.max(0,-(o*f+a)),g=-h*h+f*(f+2*c)+u;else f<=-x?(h=Math.max(0,-(-o*s+a)),f=h>0?-s:Math.min(Math.max(-s,-c),s),g=-h*h+f*(f+2*c)+u):f<=x?(h=0,f=Math.min(Math.max(-s,-c),s),g=f*(f+2*c)+u):(h=Math.max(0,-(o*s+a)),f=h>0?s:Math.min(Math.max(-s,-c),s),g=-h*h+f*(f+2*c)+u);else f=o>0?-s:s,h=Math.max(0,-(o*f+a)),g=-h*h+f*(f+2*c)+u;return i&&i.copy(this.origin).addScaledVector(this.direction,h),r&&r.copy(Fu).addScaledVector(Ka,f),g}intersectSphere(e,n){Mi.subVectors(e.center,this.origin);const i=Mi.dot(this.direction),r=Mi.dot(Mi)-i*i,s=e.radius*e.radius;if(r>s)return null;const o=Math.sqrt(s-r),a=i-o,c=i+o;return c<0?null:a<0?this.at(c,n):this.at(a,n)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const n=e.normal.dot(this.direction);if(n===0)return e.distanceToPoint(this.origin)===0?0:null;const i=-(this.origin.dot(e.normal)+e.constant)/n;return i>=0?i:null}intersectPlane(e,n){const i=this.distanceToPlane(e);return i===null?null:this.at(i,n)}intersectsPlane(e){const n=e.distanceToPoint(this.origin);return n===0||e.normal.dot(this.direction)*n<0}intersectBox(e,n){let i,r,s,o,a,c;const u=1/this.direction.x,p=1/this.direction.y,h=1/this.direction.z,f=this.origin;return u>=0?(i=(e.min.x-f.x)*u,r=(e.max.x-f.x)*u):(i=(e.max.x-f.x)*u,r=(e.min.x-f.x)*u),p>=0?(s=(e.min.y-f.y)*p,o=(e.max.y-f.y)*p):(s=(e.max.y-f.y)*p,o=(e.min.y-f.y)*p),i>o||s>r||((s>i||isNaN(i))&&(i=s),(o<r||isNaN(r))&&(r=o),h>=0?(a=(e.min.z-f.z)*h,c=(e.max.z-f.z)*h):(a=(e.max.z-f.z)*h,c=(e.min.z-f.z)*h),i>c||a>r)||((a>i||i!==i)&&(i=a),(c<r||r!==r)&&(r=c),r<0)?null:this.at(i>=0?i:r,n)}intersectsBox(e){return this.intersectBox(e,Mi)!==null}intersectTriangle(e,n,i,r,s){Nu.subVectors(n,e),Ya.subVectors(i,e),Uu.crossVectors(Nu,Ya);let o=this.direction.dot(Uu),a;if(o>0){if(r)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Wi.subVectors(this.origin,e);const c=a*this.direction.dot(Ya.crossVectors(Wi,Ya));if(c<0)return null;const u=a*this.direction.dot(Nu.cross(Wi));if(u<0||c+u>o)return null;const p=-a*Wi.dot(Uu);return p<0?null:this.at(p/o,s)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class As extends vr{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new tt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new pi,this.combine=k1,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Om=new _t,br=new Wp,Za=new pa,zm=new H,Ja=new H,Qa=new H,el=new H,ku=new H,tl=new H,Bm=new H,nl=new H;class pn extends Gt{constructor(e=new rn,n=new As){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){const r=n[i[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,o=r.length;s<o;s++){const a=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=s}}}}getVertexPosition(e,n){const i=this.geometry,r=i.attributes.position,s=i.morphAttributes.position,o=i.morphTargetsRelative;n.fromBufferAttribute(r,e);const a=this.morphTargetInfluences;if(s&&a){tl.set(0,0,0);for(let c=0,u=s.length;c<u;c++){const p=a[c],h=s[c];p!==0&&(ku.fromBufferAttribute(h,e),o?tl.addScaledVector(ku,p):tl.addScaledVector(ku.sub(n),p))}n.add(tl)}return n}raycast(e,n){const i=this.geometry,r=this.material,s=this.matrixWorld;r!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Za.copy(i.boundingSphere),Za.applyMatrix4(s),br.copy(e.ray).recast(e.near),!(Za.containsPoint(br.origin)===!1&&(br.intersectSphere(Za,zm)===null||br.origin.distanceToSquared(zm)>(e.far-e.near)**2))&&(Om.copy(s).invert(),br.copy(e.ray).applyMatrix4(Om),!(i.boundingBox!==null&&br.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,n,br)))}_computeIntersections(e,n,i){let r;const s=this.geometry,o=this.material,a=s.index,c=s.attributes.position,u=s.attributes.uv,p=s.attributes.uv1,h=s.attributes.normal,f=s.groups,g=s.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,M=f.length;x<M;x++){const v=f[x],d=o[v.materialIndex],m=Math.max(v.start,g.start),_=Math.min(a.count,Math.min(v.start+v.count,g.start+g.count));for(let b=m,w=_;b<w;b+=3){const A=a.getX(b),E=a.getX(b+1),y=a.getX(b+2);r=il(this,d,e,i,u,p,h,A,E,y),r&&(r.faceIndex=Math.floor(b/3),r.face.materialIndex=v.materialIndex,n.push(r))}}else{const x=Math.max(0,g.start),M=Math.min(a.count,g.start+g.count);for(let v=x,d=M;v<d;v+=3){const m=a.getX(v),_=a.getX(v+1),b=a.getX(v+2);r=il(this,o,e,i,u,p,h,m,_,b),r&&(r.faceIndex=Math.floor(v/3),n.push(r))}}else if(c!==void 0)if(Array.isArray(o))for(let x=0,M=f.length;x<M;x++){const v=f[x],d=o[v.materialIndex],m=Math.max(v.start,g.start),_=Math.min(c.count,Math.min(v.start+v.count,g.start+g.count));for(let b=m,w=_;b<w;b+=3){const A=b,E=b+1,y=b+2;r=il(this,d,e,i,u,p,h,A,E,y),r&&(r.faceIndex=Math.floor(b/3),r.face.materialIndex=v.materialIndex,n.push(r))}}else{const x=Math.max(0,g.start),M=Math.min(c.count,g.start+g.count);for(let v=x,d=M;v<d;v+=3){const m=v,_=v+1,b=v+2;r=il(this,o,e,i,u,p,h,m,_,b),r&&(r.faceIndex=Math.floor(v/3),n.push(r))}}}}function _S(t,e,n,i,r,s,o,a){let c;if(e.side===en?c=i.intersectTriangle(o,s,r,!0,a):c=i.intersectTriangle(r,s,o,e.side===pr,a),c===null)return null;nl.copy(a),nl.applyMatrix4(t.matrixWorld);const u=n.ray.origin.distanceTo(nl);return u<n.near||u>n.far?null:{distance:u,point:nl.clone(),object:t}}function il(t,e,n,i,r,s,o,a,c,u){t.getVertexPosition(a,Ja),t.getVertexPosition(c,Qa),t.getVertexPosition(u,el);const p=_S(t,e,n,i,Ja,Qa,el,Bm);if(p){const h=new H;Ln.getBarycoord(Bm,Ja,Qa,el,h),r&&(p.uv=Ln.getInterpolatedAttribute(r,a,c,u,h,new Ze)),s&&(p.uv1=Ln.getInterpolatedAttribute(s,a,c,u,h,new Ze)),o&&(p.normal=Ln.getInterpolatedAttribute(o,a,c,u,h,new H),p.normal.dot(i.direction)>0&&p.normal.multiplyScalar(-1));const f={a,b:c,c:u,normal:new H,materialIndex:0};Ln.getNormal(Ja,Qa,el,f.normal),p.face=f,p.barycoord=h}return p}class yS extends tn{constructor(e=null,n=1,i=1,r,s,o,a,c,u=Vt,p=Vt,h,f){super(null,o,a,c,u,p,r,s,h,f),this.isDataTexture=!0,this.image={data:e,width:n,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const Ou=new H,SS=new H,MS=new qe;class Ir{constructor(e=new H(1,0,0),n=0){this.isPlane=!0,this.normal=e,this.constant=n}set(e,n){return this.normal.copy(e),this.constant=n,this}setComponents(e,n,i,r){return this.normal.set(e,n,i),this.constant=r,this}setFromNormalAndCoplanarPoint(e,n){return this.normal.copy(e),this.constant=-n.dot(this.normal),this}setFromCoplanarPoints(e,n,i){const r=Ou.subVectors(i,n).cross(SS.subVectors(e,n)).normalize();return this.setFromNormalAndCoplanarPoint(r,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,n){return n.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,n){const i=e.delta(Ou),r=this.normal.dot(i);if(r===0)return this.distanceToPoint(e.start)===0?n.copy(e.start):null;const s=-(e.start.dot(this.normal)+this.constant)/r;return s<0||s>1?null:n.copy(e.start).addScaledVector(i,s)}intersectsLine(e){const n=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return n<0&&i>0||i<0&&n>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,n){const i=n||MS.getNormalMatrix(e),r=this.coplanarPoint(Ou).applyMatrix4(e),s=this.normal.applyMatrix3(i).normalize();return this.constant=-r.dot(s),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Er=new pa,bS=new Ze(.5,.5),rl=new H;class Xp{constructor(e=new Ir,n=new Ir,i=new Ir,r=new Ir,s=new Ir,o=new Ir){this.planes=[e,n,i,r,s,o]}set(e,n,i,r,s,o){const a=this.planes;return a[0].copy(e),a[1].copy(n),a[2].copy(i),a[3].copy(r),a[4].copy(s),a[5].copy(o),this}copy(e){const n=this.planes;for(let i=0;i<6;i++)n[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,n=ai,i=!1){const r=this.planes,s=e.elements,o=s[0],a=s[1],c=s[2],u=s[3],p=s[4],h=s[5],f=s[6],g=s[7],x=s[8],M=s[9],v=s[10],d=s[11],m=s[12],_=s[13],b=s[14],w=s[15];if(r[0].setComponents(u-o,g-p,d-x,w-m).normalize(),r[1].setComponents(u+o,g+p,d+x,w+m).normalize(),r[2].setComponents(u+a,g+h,d+M,w+_).normalize(),r[3].setComponents(u-a,g-h,d-M,w-_).normalize(),i)r[4].setComponents(c,f,v,b).normalize(),r[5].setComponents(u-c,g-f,d-v,w-b).normalize();else if(r[4].setComponents(u-c,g-f,d-v,w-b).normalize(),n===ai)r[5].setComponents(u+c,g+f,d+v,w+b).normalize();else if(n===na)r[5].setComponents(c,f,v,b).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+n);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Er.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const n=e.geometry;n.boundingSphere===null&&n.computeBoundingSphere(),Er.copy(n.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Er)}intersectsSprite(e){Er.center.set(0,0,0);const n=bS.distanceTo(e.center);return Er.radius=.7071067811865476+n,Er.applyMatrix4(e.matrixWorld),this.intersectsSphere(Er)}intersectsSphere(e){const n=this.planes,i=e.center,r=-e.radius;for(let s=0;s<6;s++)if(n[s].distanceToPoint(i)<r)return!1;return!0}intersectsBox(e){const n=this.planes;for(let i=0;i<6;i++){const r=n[i];if(rl.x=r.normal.x>0?e.max.x:e.min.x,rl.y=r.normal.y>0?e.max.y:e.min.y,rl.z=r.normal.z>0?e.max.z:e.min.z,r.distanceToPoint(rl)<0)return!1}return!0}containsPoint(e){const n=this.planes;for(let i=0;i<6;i++)if(n[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class lx extends vr{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new tt(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}}const cc=new H,uc=new H,jm=new _t,_o=new Wp,sl=new pa,zu=new H,Vm=new H;class ES extends Gt{constructor(e=new rn,n=new lx){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){const e=this.geometry;if(e.index===null){const n=e.attributes.position,i=[0];for(let r=1,s=n.count;r<s;r++)cc.fromBufferAttribute(n,r-1),uc.fromBufferAttribute(n,r),i[r]=i[r-1],i[r]+=cc.distanceTo(uc);e.setAttribute("lineDistance",new Wt(i,1))}else He("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,n){const i=this.geometry,r=this.matrixWorld,s=e.params.Line.threshold,o=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),sl.copy(i.boundingSphere),sl.applyMatrix4(r),sl.radius+=s,e.ray.intersectsSphere(sl)===!1)return;jm.copy(r).invert(),_o.copy(e.ray).applyMatrix4(jm);const a=s/((this.scale.x+this.scale.y+this.scale.z)/3),c=a*a,u=this.isLineSegments?2:1,p=i.index,f=i.attributes.position;if(p!==null){const g=Math.max(0,o.start),x=Math.min(p.count,o.start+o.count);for(let M=g,v=x-1;M<v;M+=u){const d=p.getX(M),m=p.getX(M+1),_=ol(this,e,_o,c,d,m,M);_&&n.push(_)}if(this.isLineLoop){const M=p.getX(x-1),v=p.getX(g),d=ol(this,e,_o,c,M,v,x-1);d&&n.push(d)}}else{const g=Math.max(0,o.start),x=Math.min(f.count,o.start+o.count);for(let M=g,v=x-1;M<v;M+=u){const d=ol(this,e,_o,c,M,M+1,M);d&&n.push(d)}if(this.isLineLoop){const M=ol(this,e,_o,c,x-1,g,x-1);M&&n.push(M)}}}updateMorphTargets(){const n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){const r=n[i[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,o=r.length;s<o;s++){const a=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=s}}}}}function ol(t,e,n,i,r,s,o){const a=t.geometry.attributes.position;if(cc.fromBufferAttribute(a,r),uc.fromBufferAttribute(a,s),n.distanceSqToSegment(cc,uc,zu,Vm)>i)return;zu.applyMatrix4(t.matrixWorld);const u=e.ray.origin.distanceTo(zu);if(!(u<e.near||u>e.far))return{distance:u,point:Vm.clone().applyMatrix4(t.matrixWorld),index:o,face:null,faceIndex:null,barycoord:null,object:t}}class cx extends vr{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new tt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const Hm=new _t,Lf=new Wp,al=new pa,ll=new H;class TS extends Gt{constructor(e=new rn,n=new cx){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,n){const i=this.geometry,r=this.matrixWorld,s=e.params.Points.threshold,o=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),al.copy(i.boundingSphere),al.applyMatrix4(r),al.radius+=s,e.ray.intersectsSphere(al)===!1)return;Hm.copy(r).invert(),Lf.copy(e.ray).applyMatrix4(Hm);const a=s/((this.scale.x+this.scale.y+this.scale.z)/3),c=a*a,u=i.index,h=i.attributes.position;if(u!==null){const f=Math.max(0,o.start),g=Math.min(u.count,o.start+o.count);for(let x=f,M=g;x<M;x++){const v=u.getX(x);ll.fromBufferAttribute(h,v),Gm(ll,v,c,r,e,n,this)}}else{const f=Math.max(0,o.start),g=Math.min(h.count,o.start+o.count);for(let x=f,M=g;x<M;x++)ll.fromBufferAttribute(h,x),Gm(ll,x,c,r,e,n,this)}}updateMorphTargets(){const n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){const r=n[i[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,o=r.length;s<o;s++){const a=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=s}}}}}function Gm(t,e,n,i,r,s,o){const a=Lf.distanceSqToPoint(t);if(a<n){const c=new H;Lf.closestPointToPoint(t,c),c.applyMatrix4(i);const u=r.ray.origin.distanceTo(c);if(u<r.near||u>r.far)return;s.push({distance:u,distanceToRay:Math.sqrt(a),point:c,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class ux extends tn{constructor(e=[],n=Xr,i,r,s,o,a,c,u,p){super(e,n,i,r,s,o,a,c,u,p),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class wS extends tn{constructor(e,n,i,r,s,o,a,c,u){super(e,n,i,r,s,o,a,c,u),this.isCanvasTexture=!0,this.needsUpdate=!0}}class ia extends tn{constructor(e,n,i=fi,r,s,o,a=Vt,c=Vt,u,p=Ui,h=1){if(p!==Ui&&p!==kr)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const f={width:e,height:n,depth:h};super(f,r,s,o,a,c,p,i,u),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Gp(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const n=super.toJSON(e);return this.compareFunction!==null&&(n.compareFunction=this.compareFunction),n}}class CS extends ia{constructor(e,n=fi,i=Xr,r,s,o=Vt,a=Vt,c,u=Ui){const p={width:e,height:e,depth:1},h=[p,p,p,p,p,p];super(e,e,n,i,r,s,o,a,c,u),this.image=h,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}}class dx extends tn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class ha extends rn{constructor(e=1,n=1,i=1,r=1,s=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:n,depth:i,widthSegments:r,heightSegments:s,depthSegments:o};const a=this;r=Math.floor(r),s=Math.floor(s),o=Math.floor(o);const c=[],u=[],p=[],h=[];let f=0,g=0;x("z","y","x",-1,-1,i,n,e,o,s,0),x("z","y","x",1,-1,i,n,-e,o,s,1),x("x","z","y",1,1,e,i,n,r,o,2),x("x","z","y",1,-1,e,i,-n,r,o,3),x("x","y","z",1,-1,e,n,i,r,s,4),x("x","y","z",-1,-1,e,n,-i,r,s,5),this.setIndex(c),this.setAttribute("position",new Wt(u,3)),this.setAttribute("normal",new Wt(p,3)),this.setAttribute("uv",new Wt(h,2));function x(M,v,d,m,_,b,w,A,E,y,C){const P=b/E,I=w/y,F=b/2,B=w/2,W=A/2,V=E+1,G=y+1;let z=0,j=0;const $=new H;for(let Q=0;Q<G;Q++){const se=Q*I-B;for(let ae=0;ae<V;ae++){const Ae=ae*P-F;$[M]=Ae*m,$[v]=se*_,$[d]=W,u.push($.x,$.y,$.z),$[M]=0,$[v]=0,$[d]=A>0?1:-1,p.push($.x,$.y,$.z),h.push(ae/E),h.push(1-Q/y),z+=1}}for(let Q=0;Q<y;Q++)for(let se=0;se<E;se++){const ae=f+se+V*Q,Ae=f+se+V*(Q+1),De=f+(se+1)+V*(Q+1),Oe=f+(se+1)+V*Q;c.push(ae,Ae,Oe),c.push(Ae,De,Oe),j+=6}a.addGroup(g,j,C),g+=j,f+=z}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ha(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}class Ic extends rn{constructor(e=1,n=1,i=1,r=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:n,widthSegments:i,heightSegments:r};const s=e/2,o=n/2,a=Math.floor(i),c=Math.floor(r),u=a+1,p=c+1,h=e/a,f=n/c,g=[],x=[],M=[],v=[];for(let d=0;d<p;d++){const m=d*f-o;for(let _=0;_<u;_++){const b=_*h-s;x.push(b,-m,0),M.push(0,0,1),v.push(_/a),v.push(1-d/c)}}for(let d=0;d<c;d++)for(let m=0;m<a;m++){const _=m+u*d,b=m+u*(d+1),w=m+1+u*(d+1),A=m+1+u*d;g.push(_,b,A),g.push(b,w,A)}this.setIndex(g),this.setAttribute("position",new Wt(x,3)),this.setAttribute("normal",new Wt(M,3)),this.setAttribute("uv",new Wt(v,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Ic(e.width,e.height,e.widthSegments,e.heightSegments)}}class qp extends rn{constructor(e=.5,n=1,i=32,r=1,s=0,o=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:e,outerRadius:n,thetaSegments:i,phiSegments:r,thetaStart:s,thetaLength:o},i=Math.max(3,i),r=Math.max(1,r);const a=[],c=[],u=[],p=[];let h=e;const f=(n-e)/r,g=new H,x=new Ze;for(let M=0;M<=r;M++){for(let v=0;v<=i;v++){const d=s+v/i*o;g.x=h*Math.cos(d),g.y=h*Math.sin(d),c.push(g.x,g.y,g.z),u.push(0,0,1),x.x=(g.x/n+1)/2,x.y=(g.y/n+1)/2,p.push(x.x,x.y)}h+=f}for(let M=0;M<r;M++){const v=M*(i+1);for(let d=0;d<i;d++){const m=d+v,_=m,b=m+i+1,w=m+i+2,A=m+1;a.push(_,b,A),a.push(b,w,A)}}this.setIndex(a),this.setAttribute("position",new Wt(c,3)),this.setAttribute("normal",new Wt(u,3)),this.setAttribute("uv",new Wt(p,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new qp(e.innerRadius,e.outerRadius,e.thetaSegments,e.phiSegments,e.thetaStart,e.thetaLength)}}class Rs extends rn{constructor(e=1,n=32,i=16,r=0,s=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:n,heightSegments:i,phiStart:r,phiLength:s,thetaStart:o,thetaLength:a},n=Math.max(3,Math.floor(n)),i=Math.max(2,Math.floor(i));const c=Math.min(o+a,Math.PI);let u=0;const p=[],h=new H,f=new H,g=[],x=[],M=[],v=[];for(let d=0;d<=i;d++){const m=[],_=d/i;let b=0;d===0&&o===0?b=.5/n:d===i&&c===Math.PI&&(b=-.5/n);for(let w=0;w<=n;w++){const A=w/n;h.x=-e*Math.cos(r+A*s)*Math.sin(o+_*a),h.y=e*Math.cos(o+_*a),h.z=e*Math.sin(r+A*s)*Math.sin(o+_*a),x.push(h.x,h.y,h.z),f.copy(h).normalize(),M.push(f.x,f.y,f.z),v.push(A+b,1-_),m.push(u++)}p.push(m)}for(let d=0;d<i;d++)for(let m=0;m<n;m++){const _=p[d][m+1],b=p[d][m],w=p[d+1][m],A=p[d+1][m+1];(d!==0||o>0)&&g.push(_,b,A),(d!==i-1||c<Math.PI)&&g.push(b,w,A)}this.setIndex(g),this.setAttribute("position",new Wt(x,3)),this.setAttribute("normal",new Wt(M,3)),this.setAttribute("uv",new Wt(v,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Rs(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}function Ks(t){const e={};for(const n in t){e[n]={};for(const i in t[n]){const r=t[n][i];r&&(r.isColor||r.isMatrix3||r.isMatrix4||r.isVector2||r.isVector3||r.isVector4||r.isTexture||r.isQuaternion)?r.isRenderTargetTexture?(He("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[n][i]=null):e[n][i]=r.clone():Array.isArray(r)?e[n][i]=r.slice():e[n][i]=r}}return e}function on(t){const e={};for(let n=0;n<t.length;n++){const i=Ks(t[n]);for(const r in i)e[r]=i[r]}return e}function AS(t){const e=[];for(let n=0;n<t.length;n++)e.push(t[n].clone());return e}function fx(t){const e=t.getRenderTarget();return e===null?t.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:rt.workingColorSpace}const RS={clone:Ks,merge:on};var IS=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,PS=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class hi extends vr{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=IS,this.fragmentShader=PS,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Ks(e.uniforms),this.uniformsGroups=AS(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){const n=super.toJSON(e);n.glslVersion=this.glslVersion,n.uniforms={};for(const r in this.uniforms){const o=this.uniforms[r].value;o&&o.isTexture?n.uniforms[r]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?n.uniforms[r]={type:"c",value:o.getHex()}:o&&o.isVector2?n.uniforms[r]={type:"v2",value:o.toArray()}:o&&o.isVector3?n.uniforms[r]={type:"v3",value:o.toArray()}:o&&o.isVector4?n.uniforms[r]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?n.uniforms[r]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?n.uniforms[r]={type:"m4",value:o.toArray()}:n.uniforms[r]={value:o}}Object.keys(this.defines).length>0&&(n.defines=this.defines),n.vertexShader=this.vertexShader,n.fragmentShader=this.fragmentShader,n.lights=this.lights,n.clipping=this.clipping;const i={};for(const r in this.extensions)this.extensions[r]===!0&&(i[r]=!0);return Object.keys(i).length>0&&(n.extensions=i),n}}class DS extends hi{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}}class LS extends vr{constructor(e){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new tt(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new tt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Q1,this.normalScale=new Ze(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new pi,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.defines={STANDARD:""},this.color.copy(e.color),this.roughness=e.roughness,this.metalness=e.metalness,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.roughnessMap=e.roughnessMap,this.metalnessMap=e.metalnessMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.envMapIntensity=e.envMapIntensity,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}}class FS extends vr{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=By,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class NS extends vr{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class px extends Gt{constructor(e,n=1){super(),this.isLight=!0,this.type="Light",this.color=new tt(e),this.intensity=n}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,n){return super.copy(e,n),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){const n=super.toJSON(e);return n.object.color=this.color.getHex(),n.object.intensity=this.intensity,n}}const Bu=new _t,Wm=new H,Xm=new H;class US{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new Ze(512,512),this.mapType=bn,this.map=null,this.mapPass=null,this.matrix=new _t,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Xp,this._frameExtents=new Ze(1,1),this._viewportCount=1,this._viewports=[new wt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){const n=this.camera,i=this.matrix;Wm.setFromMatrixPosition(e.matrixWorld),n.position.copy(Wm),Xm.setFromMatrixPosition(e.target.matrixWorld),n.lookAt(Xm),n.updateMatrixWorld(),Bu.multiplyMatrices(n.projectionMatrix,n.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Bu,n.coordinateSystem,n.reversedDepth),n.coordinateSystem===na||n.reversedDepth?i.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(Bu)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){const e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}}const cl=new H,ul=new to,Qn=new H;class hx extends Gt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new _t,this.projectionMatrix=new _t,this.projectionMatrixInverse=new _t,this.coordinateSystem=ai,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,n){return super.copy(e,n),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(cl,ul,Qn),Qn.x===1&&Qn.y===1&&Qn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(cl,ul,Qn.set(1,1,1)).invert()}updateWorldMatrix(e,n){super.updateWorldMatrix(e,n),this.matrixWorld.decompose(cl,ul,Qn),Qn.x===1&&Qn.y===1&&Qn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(cl,ul,Qn.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}}const Xi=new H,qm=new Ze,$m=new Ze;class Mn extends hx{constructor(e=50,n=1,i=.1,r=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=r,this.focus=10,this.aspect=n,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const n=.5*this.getFilmHeight()/e;this.fov=Df*2*Math.atan(n),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(hu*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return Df*2*Math.atan(Math.tan(hu*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,n,i){Xi.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Xi.x,Xi.y).multiplyScalar(-e/Xi.z),Xi.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(Xi.x,Xi.y).multiplyScalar(-e/Xi.z)}getViewSize(e,n){return this.getViewBounds(e,qm,$m),n.subVectors($m,qm)}setViewOffset(e,n,i,r,s,o){this.aspect=e/n,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=r,this.view.width=s,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let n=e*Math.tan(hu*.5*this.fov)/this.zoom,i=2*n,r=this.aspect*i,s=-.5*r;const o=this.view;if(this.view!==null&&this.view.enabled){const c=o.fullWidth,u=o.fullHeight;s+=o.offsetX*r/c,n-=o.offsetY*i/u,r*=o.width/c,i*=o.height/u}const a=this.filmOffset;a!==0&&(s+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(s,s+r,n,n-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const n=super.toJSON(e);return n.object.fov=this.fov,n.object.zoom=this.zoom,n.object.near=this.near,n.object.far=this.far,n.object.focus=this.focus,n.object.aspect=this.aspect,this.view!==null&&(n.object.view=Object.assign({},this.view)),n.object.filmGauge=this.filmGauge,n.object.filmOffset=this.filmOffset,n}}class kS extends US{constructor(){super(new Mn(90,1,.5,500)),this.isPointLightShadow=!0}}class OS extends px{constructor(e,n,i=0,r=2){super(e,n),this.isPointLight=!0,this.type="PointLight",this.distance=i,this.decay=r,this.shadow=new kS}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){super.dispose(),this.shadow.dispose()}copy(e,n){return super.copy(e,n),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}toJSON(e){const n=super.toJSON(e);return n.object.distance=this.distance,n.object.decay=this.decay,n.object.shadow=this.shadow.toJSON(),n}}class mx extends hx{constructor(e=-1,n=1,i=1,r=-1,s=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=n,this.top=i,this.bottom=r,this.near=s,this.far=o,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,n,i,r,s,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=r,this.view.width=s,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),n=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,r=(this.top+this.bottom)/2;let s=i-e,o=i+e,a=r+n,c=r-n;if(this.view!==null&&this.view.enabled){const u=(this.right-this.left)/this.view.fullWidth/this.zoom,p=(this.top-this.bottom)/this.view.fullHeight/this.zoom;s+=u*this.view.offsetX,o=s+u*this.view.width,a-=p*this.view.offsetY,c=a-p*this.view.height}this.projectionMatrix.makeOrthographic(s,o,a,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const n=super.toJSON(e);return n.object.zoom=this.zoom,n.object.left=this.left,n.object.right=this.right,n.object.top=this.top,n.object.bottom=this.bottom,n.object.near=this.near,n.object.far=this.far,this.view!==null&&(n.object.view=Object.assign({},this.view)),n}}class zS extends px{constructor(e,n){super(e,n),this.isAmbientLight=!0,this.type="AmbientLight"}}const ps=-90,hs=1;class BS extends Gt{constructor(e,n,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;const r=new Mn(ps,hs,e,n);r.layers=this.layers,this.add(r);const s=new Mn(ps,hs,e,n);s.layers=this.layers,this.add(s);const o=new Mn(ps,hs,e,n);o.layers=this.layers,this.add(o);const a=new Mn(ps,hs,e,n);a.layers=this.layers,this.add(a);const c=new Mn(ps,hs,e,n);c.layers=this.layers,this.add(c);const u=new Mn(ps,hs,e,n);u.layers=this.layers,this.add(u)}updateCoordinateSystem(){const e=this.coordinateSystem,n=this.children.concat(),[i,r,s,o,a,c]=n;for(const u of n)this.remove(u);if(e===ai)i.up.set(0,1,0),i.lookAt(1,0,0),r.up.set(0,1,0),r.lookAt(-1,0,0),s.up.set(0,0,-1),s.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(e===na)i.up.set(0,-1,0),i.lookAt(-1,0,0),r.up.set(0,-1,0),r.lookAt(1,0,0),s.up.set(0,0,1),s.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const u of n)this.add(u),u.updateMatrixWorld()}update(e,n){this.parent===null&&this.updateMatrixWorld();const{renderTarget:i,activeMipmapLevel:r}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[s,o,a,c,u,p]=this.children,h=e.getRenderTarget(),f=e.getActiveCubeFace(),g=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const M=i.texture.generateMipmaps;i.texture.generateMipmaps=!1;let v=!1;e.isWebGLRenderer===!0?v=e.state.buffers.depth.getReversed():v=e.reversedDepthBuffer,e.setRenderTarget(i,0,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,s),e.setRenderTarget(i,1,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,o),e.setRenderTarget(i,2,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,a),e.setRenderTarget(i,3,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,c),e.setRenderTarget(i,4,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,u),i.texture.generateMipmaps=M,e.setRenderTarget(i,5,r),v&&e.autoClear===!1&&e.clearDepth(),e.render(n,p),e.setRenderTarget(h,f,g),e.xr.enabled=x,i.texture.needsPMREMUpdate=!0}}class jS extends Mn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}let VS=class{constructor(e=!0){this.autoStart=e,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1,He("THREE.Clock: This module has been deprecated. Please use THREE.Timer instead.")}start(){this.startTime=performance.now(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let e=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){const n=performance.now();e=(n-this.oldTime)/1e3,this.oldTime=n,this.elapsedTime+=e}return e}};function Km(t,e,n,i){const r=HS(i);switch(n){case Y1:return t*e;case J1:return t*e/r.components*r.byteLength;case zp:return t*e/r.components*r.byteLength;case qs:return t*e*2/r.components*r.byteLength;case Bp:return t*e*2/r.components*r.byteLength;case Z1:return t*e*3/r.components*r.byteLength;case Xn:return t*e*4/r.components*r.byteLength;case jp:return t*e*4/r.components*r.byteLength;case Cl:case Al:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case Rl:case Il:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case Qd:case tf:return Math.max(t,16)*Math.max(e,8)/4;case Jd:case ef:return Math.max(t,8)*Math.max(e,8)/2;case nf:case rf:case of:case af:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case sf:case lf:case cf:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case uf:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case df:return Math.floor((t+4)/5)*Math.floor((e+3)/4)*16;case ff:return Math.floor((t+4)/5)*Math.floor((e+4)/5)*16;case pf:return Math.floor((t+5)/6)*Math.floor((e+4)/5)*16;case hf:return Math.floor((t+5)/6)*Math.floor((e+5)/6)*16;case mf:return Math.floor((t+7)/8)*Math.floor((e+4)/5)*16;case gf:return Math.floor((t+7)/8)*Math.floor((e+5)/6)*16;case xf:return Math.floor((t+7)/8)*Math.floor((e+7)/8)*16;case vf:return Math.floor((t+9)/10)*Math.floor((e+4)/5)*16;case _f:return Math.floor((t+9)/10)*Math.floor((e+5)/6)*16;case yf:return Math.floor((t+9)/10)*Math.floor((e+7)/8)*16;case Sf:return Math.floor((t+9)/10)*Math.floor((e+9)/10)*16;case Mf:return Math.floor((t+11)/12)*Math.floor((e+9)/10)*16;case bf:return Math.floor((t+11)/12)*Math.floor((e+11)/12)*16;case Ef:case Tf:case wf:return Math.ceil(t/4)*Math.ceil(e/4)*16;case Cf:case Af:return Math.ceil(t/4)*Math.ceil(e/4)*8;case Rf:case If:return Math.ceil(t/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${n} format.`)}function HS(t){switch(t){case bn:case X1:return{byteLength:1,components:1};case ea:case q1:case Ni:return{byteLength:2,components:1};case kp:case Op:return{byteLength:2,components:4};case fi:case Up:case oi:return{byteLength:4,components:1};case $1:case K1:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${t}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Np}}));typeof window<"u"&&(window.__THREE__?He("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Np);/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function gx(){let t=null,e=!1,n=null,i=null;function r(s,o){n(s,o),i=t.requestAnimationFrame(r)}return{start:function(){e!==!0&&n!==null&&(i=t.requestAnimationFrame(r),e=!0)},stop:function(){t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(s){n=s},setContext:function(s){t=s}}}function GS(t){const e=new WeakMap;function n(a,c){const u=a.array,p=a.usage,h=u.byteLength,f=t.createBuffer();t.bindBuffer(c,f),t.bufferData(c,u,p),a.onUploadCallback();let g;if(u instanceof Float32Array)g=t.FLOAT;else if(typeof Float16Array<"u"&&u instanceof Float16Array)g=t.HALF_FLOAT;else if(u instanceof Uint16Array)a.isFloat16BufferAttribute?g=t.HALF_FLOAT:g=t.UNSIGNED_SHORT;else if(u instanceof Int16Array)g=t.SHORT;else if(u instanceof Uint32Array)g=t.UNSIGNED_INT;else if(u instanceof Int32Array)g=t.INT;else if(u instanceof Int8Array)g=t.BYTE;else if(u instanceof Uint8Array)g=t.UNSIGNED_BYTE;else if(u instanceof Uint8ClampedArray)g=t.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+u);return{buffer:f,type:g,bytesPerElement:u.BYTES_PER_ELEMENT,version:a.version,size:h}}function i(a,c,u){const p=c.array,h=c.updateRanges;if(t.bindBuffer(u,a),h.length===0)t.bufferSubData(u,0,p);else{h.sort((g,x)=>g.start-x.start);let f=0;for(let g=1;g<h.length;g++){const x=h[f],M=h[g];M.start<=x.start+x.count+1?x.count=Math.max(x.count,M.start+M.count-x.start):(++f,h[f]=M)}h.length=f+1;for(let g=0,x=h.length;g<x;g++){const M=h[g];t.bufferSubData(u,M.start*p.BYTES_PER_ELEMENT,p,M.start,M.count)}c.clearUpdateRanges()}c.onUploadCallback()}function r(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function s(a){a.isInterleavedBufferAttribute&&(a=a.data);const c=e.get(a);c&&(t.deleteBuffer(c.buffer),e.delete(a))}function o(a,c){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const p=e.get(a);(!p||p.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const u=e.get(a);if(u===void 0)e.set(a,n(a,c));else if(u.version<a.version){if(u.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(u.buffer,a,c),u.version=a.version}}return{get:r,remove:s,update:o}}var WS=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,XS=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,qS=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,$S=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,KS=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,YS=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,ZS=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,JS=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,QS=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,eM=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,tM=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,nM=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,iM=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,rM=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,sM=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,oM=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,aM=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,lM=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,cM=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,uM=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,dM=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,fM=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,pM=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,hM=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,mM=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,gM=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,xM=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,vM=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,_M=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,yM=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,SM="gl_FragColor = linearToOutputTexel( gl_FragColor );",MM=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,bM=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,EM=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,TM=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,wM=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,CM=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,AM=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,RM=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,IM=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,PM=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,DM=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,LM=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,FM=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,NM=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,UM=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,kM=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,OM=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,zM=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,BM=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,jM=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,VM=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,HM=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return v;
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,GM=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,WM=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,XM=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,qM=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,$M=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,KM=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,YM=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,ZM=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,JM=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,QM=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,eb=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,tb=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,nb=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,ib=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,rb=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,sb=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,ob=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,ab=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,lb=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,cb=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,ub=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,db=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,fb=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,pb=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,hb=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,mb=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,gb=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,xb=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,vb=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,_b=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,yb=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Sb=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Mb=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,bb=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,Eb=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,Tb=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,wb=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,Cb=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,Ab=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,Rb=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,Ib=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,Pb=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,Db=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,Lb=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,Fb=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Nb=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Ub=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,kb=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Ob=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,zb=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Bb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,jb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Vb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Hb=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Gb=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Wb=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Xb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,qb=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,$b=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Kb=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Yb=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Zb=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Jb=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Qb=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,e2=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,t2=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,n2=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,i2=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,r2=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,s2=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,o2=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,a2=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,l2=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,c2=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,u2=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,d2=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,f2=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,p2=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,h2=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,m2=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,g2=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,x2=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,v2=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,_2=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,y2=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,S2=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,M2=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,b2=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Ke={alphahash_fragment:WS,alphahash_pars_fragment:XS,alphamap_fragment:qS,alphamap_pars_fragment:$S,alphatest_fragment:KS,alphatest_pars_fragment:YS,aomap_fragment:ZS,aomap_pars_fragment:JS,batching_pars_vertex:QS,batching_vertex:eM,begin_vertex:tM,beginnormal_vertex:nM,bsdfs:iM,iridescence_fragment:rM,bumpmap_pars_fragment:sM,clipping_planes_fragment:oM,clipping_planes_pars_fragment:aM,clipping_planes_pars_vertex:lM,clipping_planes_vertex:cM,color_fragment:uM,color_pars_fragment:dM,color_pars_vertex:fM,color_vertex:pM,common:hM,cube_uv_reflection_fragment:mM,defaultnormal_vertex:gM,displacementmap_pars_vertex:xM,displacementmap_vertex:vM,emissivemap_fragment:_M,emissivemap_pars_fragment:yM,colorspace_fragment:SM,colorspace_pars_fragment:MM,envmap_fragment:bM,envmap_common_pars_fragment:EM,envmap_pars_fragment:TM,envmap_pars_vertex:wM,envmap_physical_pars_fragment:kM,envmap_vertex:CM,fog_vertex:AM,fog_pars_vertex:RM,fog_fragment:IM,fog_pars_fragment:PM,gradientmap_pars_fragment:DM,lightmap_pars_fragment:LM,lights_lambert_fragment:FM,lights_lambert_pars_fragment:NM,lights_pars_begin:UM,lights_toon_fragment:OM,lights_toon_pars_fragment:zM,lights_phong_fragment:BM,lights_phong_pars_fragment:jM,lights_physical_fragment:VM,lights_physical_pars_fragment:HM,lights_fragment_begin:GM,lights_fragment_maps:WM,lights_fragment_end:XM,logdepthbuf_fragment:qM,logdepthbuf_pars_fragment:$M,logdepthbuf_pars_vertex:KM,logdepthbuf_vertex:YM,map_fragment:ZM,map_pars_fragment:JM,map_particle_fragment:QM,map_particle_pars_fragment:eb,metalnessmap_fragment:tb,metalnessmap_pars_fragment:nb,morphinstance_vertex:ib,morphcolor_vertex:rb,morphnormal_vertex:sb,morphtarget_pars_vertex:ob,morphtarget_vertex:ab,normal_fragment_begin:lb,normal_fragment_maps:cb,normal_pars_fragment:ub,normal_pars_vertex:db,normal_vertex:fb,normalmap_pars_fragment:pb,clearcoat_normal_fragment_begin:hb,clearcoat_normal_fragment_maps:mb,clearcoat_pars_fragment:gb,iridescence_pars_fragment:xb,opaque_fragment:vb,packing:_b,premultiplied_alpha_fragment:yb,project_vertex:Sb,dithering_fragment:Mb,dithering_pars_fragment:bb,roughnessmap_fragment:Eb,roughnessmap_pars_fragment:Tb,shadowmap_pars_fragment:wb,shadowmap_pars_vertex:Cb,shadowmap_vertex:Ab,shadowmask_pars_fragment:Rb,skinbase_vertex:Ib,skinning_pars_vertex:Pb,skinning_vertex:Db,skinnormal_vertex:Lb,specularmap_fragment:Fb,specularmap_pars_fragment:Nb,tonemapping_fragment:Ub,tonemapping_pars_fragment:kb,transmission_fragment:Ob,transmission_pars_fragment:zb,uv_pars_fragment:Bb,uv_pars_vertex:jb,uv_vertex:Vb,worldpos_vertex:Hb,background_vert:Gb,background_frag:Wb,backgroundCube_vert:Xb,backgroundCube_frag:qb,cube_vert:$b,cube_frag:Kb,depth_vert:Yb,depth_frag:Zb,distance_vert:Jb,distance_frag:Qb,equirect_vert:e2,equirect_frag:t2,linedashed_vert:n2,linedashed_frag:i2,meshbasic_vert:r2,meshbasic_frag:s2,meshlambert_vert:o2,meshlambert_frag:a2,meshmatcap_vert:l2,meshmatcap_frag:c2,meshnormal_vert:u2,meshnormal_frag:d2,meshphong_vert:f2,meshphong_frag:p2,meshphysical_vert:h2,meshphysical_frag:m2,meshtoon_vert:g2,meshtoon_frag:x2,points_vert:v2,points_frag:_2,shadow_vert:y2,shadow_frag:S2,sprite_vert:M2,sprite_frag:b2},_e={common:{diffuse:{value:new tt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new qe},alphaMap:{value:null},alphaMapTransform:{value:new qe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new qe}},envmap:{envMap:{value:null},envMapRotation:{value:new qe},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new qe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new qe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new qe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new qe},normalScale:{value:new Ze(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new qe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new qe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new qe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new qe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new tt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new tt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new qe},alphaTest:{value:0},uvTransform:{value:new qe}},sprite:{diffuse:{value:new tt(16777215)},opacity:{value:1},center:{value:new Ze(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new qe},alphaMap:{value:null},alphaMapTransform:{value:new qe},alphaTest:{value:0}}},ni={basic:{uniforms:on([_e.common,_e.specularmap,_e.envmap,_e.aomap,_e.lightmap,_e.fog]),vertexShader:Ke.meshbasic_vert,fragmentShader:Ke.meshbasic_frag},lambert:{uniforms:on([_e.common,_e.specularmap,_e.envmap,_e.aomap,_e.lightmap,_e.emissivemap,_e.bumpmap,_e.normalmap,_e.displacementmap,_e.fog,_e.lights,{emissive:{value:new tt(0)},envMapIntensity:{value:1}}]),vertexShader:Ke.meshlambert_vert,fragmentShader:Ke.meshlambert_frag},phong:{uniforms:on([_e.common,_e.specularmap,_e.envmap,_e.aomap,_e.lightmap,_e.emissivemap,_e.bumpmap,_e.normalmap,_e.displacementmap,_e.fog,_e.lights,{emissive:{value:new tt(0)},specular:{value:new tt(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:Ke.meshphong_vert,fragmentShader:Ke.meshphong_frag},standard:{uniforms:on([_e.common,_e.envmap,_e.aomap,_e.lightmap,_e.emissivemap,_e.bumpmap,_e.normalmap,_e.displacementmap,_e.roughnessmap,_e.metalnessmap,_e.fog,_e.lights,{emissive:{value:new tt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Ke.meshphysical_vert,fragmentShader:Ke.meshphysical_frag},toon:{uniforms:on([_e.common,_e.aomap,_e.lightmap,_e.emissivemap,_e.bumpmap,_e.normalmap,_e.displacementmap,_e.gradientmap,_e.fog,_e.lights,{emissive:{value:new tt(0)}}]),vertexShader:Ke.meshtoon_vert,fragmentShader:Ke.meshtoon_frag},matcap:{uniforms:on([_e.common,_e.bumpmap,_e.normalmap,_e.displacementmap,_e.fog,{matcap:{value:null}}]),vertexShader:Ke.meshmatcap_vert,fragmentShader:Ke.meshmatcap_frag},points:{uniforms:on([_e.points,_e.fog]),vertexShader:Ke.points_vert,fragmentShader:Ke.points_frag},dashed:{uniforms:on([_e.common,_e.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Ke.linedashed_vert,fragmentShader:Ke.linedashed_frag},depth:{uniforms:on([_e.common,_e.displacementmap]),vertexShader:Ke.depth_vert,fragmentShader:Ke.depth_frag},normal:{uniforms:on([_e.common,_e.bumpmap,_e.normalmap,_e.displacementmap,{opacity:{value:1}}]),vertexShader:Ke.meshnormal_vert,fragmentShader:Ke.meshnormal_frag},sprite:{uniforms:on([_e.sprite,_e.fog]),vertexShader:Ke.sprite_vert,fragmentShader:Ke.sprite_frag},background:{uniforms:{uvTransform:{value:new qe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Ke.background_vert,fragmentShader:Ke.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new qe}},vertexShader:Ke.backgroundCube_vert,fragmentShader:Ke.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Ke.cube_vert,fragmentShader:Ke.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Ke.equirect_vert,fragmentShader:Ke.equirect_frag},distance:{uniforms:on([_e.common,_e.displacementmap,{referencePosition:{value:new H},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Ke.distance_vert,fragmentShader:Ke.distance_frag},shadow:{uniforms:on([_e.lights,_e.fog,{color:{value:new tt(0)},opacity:{value:1}}]),vertexShader:Ke.shadow_vert,fragmentShader:Ke.shadow_frag}};ni.physical={uniforms:on([ni.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new qe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new qe},clearcoatNormalScale:{value:new Ze(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new qe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new qe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new qe},sheen:{value:0},sheenColor:{value:new tt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new qe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new qe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new qe},transmissionSamplerSize:{value:new Ze},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new qe},attenuationDistance:{value:0},attenuationColor:{value:new tt(0)},specularColor:{value:new tt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new qe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new qe},anisotropyVector:{value:new Ze},anisotropyMap:{value:null},anisotropyMapTransform:{value:new qe}}]),vertexShader:Ke.meshphysical_vert,fragmentShader:Ke.meshphysical_frag};const dl={r:0,b:0,g:0},Tr=new pi,E2=new _t;function T2(t,e,n,i,r,s){const o=new tt(0);let a=r===!0?0:1,c,u,p=null,h=0,f=null;function g(m){let _=m.isScene===!0?m.background:null;if(_&&_.isTexture){const b=m.backgroundBlurriness>0;_=e.get(_,b)}return _}function x(m){let _=!1;const b=g(m);b===null?v(o,a):b&&b.isColor&&(v(b,1),_=!0);const w=t.xr.getEnvironmentBlendMode();w==="additive"?n.buffers.color.setClear(0,0,0,1,s):w==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,s),(t.autoClear||_)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil))}function M(m,_){const b=g(_);b&&(b.isCubeTexture||b.mapping===Rc)?(u===void 0&&(u=new pn(new ha(1,1,1),new hi({name:"BackgroundCubeMaterial",uniforms:Ks(ni.backgroundCube.uniforms),vertexShader:ni.backgroundCube.vertexShader,fragmentShader:ni.backgroundCube.fragmentShader,side:en,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(w,A,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(u)),Tr.copy(_.backgroundRotation),Tr.x*=-1,Tr.y*=-1,Tr.z*=-1,b.isCubeTexture&&b.isRenderTargetTexture===!1&&(Tr.y*=-1,Tr.z*=-1),u.material.uniforms.envMap.value=b,u.material.uniforms.flipEnvMap.value=b.isCubeTexture&&b.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=_.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=_.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(E2.makeRotationFromEuler(Tr)),u.material.toneMapped=rt.getTransfer(b.colorSpace)!==lt,(p!==b||h!==b.version||f!==t.toneMapping)&&(u.material.needsUpdate=!0,p=b,h=b.version,f=t.toneMapping),u.layers.enableAll(),m.unshift(u,u.geometry,u.material,0,0,null)):b&&b.isTexture&&(c===void 0&&(c=new pn(new Ic(2,2),new hi({name:"BackgroundMaterial",uniforms:Ks(ni.background.uniforms),vertexShader:ni.background.vertexShader,fragmentShader:ni.background.fragmentShader,side:pr,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(c)),c.material.uniforms.t2D.value=b,c.material.uniforms.backgroundIntensity.value=_.backgroundIntensity,c.material.toneMapped=rt.getTransfer(b.colorSpace)!==lt,b.matrixAutoUpdate===!0&&b.updateMatrix(),c.material.uniforms.uvTransform.value.copy(b.matrix),(p!==b||h!==b.version||f!==t.toneMapping)&&(c.material.needsUpdate=!0,p=b,h=b.version,f=t.toneMapping),c.layers.enableAll(),m.unshift(c,c.geometry,c.material,0,0,null))}function v(m,_){m.getRGB(dl,fx(t)),n.buffers.color.setClear(dl.r,dl.g,dl.b,_,s)}function d(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return o},setClearColor:function(m,_=1){o.set(m),a=_,v(o,a)},getClearAlpha:function(){return a},setClearAlpha:function(m){a=m,v(o,a)},render:x,addToRenderList:M,dispose:d}}function w2(t,e){const n=t.getParameter(t.MAX_VERTEX_ATTRIBS),i={},r=f(null);let s=r,o=!1;function a(I,F,B,W,V){let G=!1;const z=h(I,W,B,F);s!==z&&(s=z,u(s.object)),G=g(I,W,B,V),G&&x(I,W,B,V),V!==null&&e.update(V,t.ELEMENT_ARRAY_BUFFER),(G||o)&&(o=!1,b(I,F,B,W),V!==null&&t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,e.get(V).buffer))}function c(){return t.createVertexArray()}function u(I){return t.bindVertexArray(I)}function p(I){return t.deleteVertexArray(I)}function h(I,F,B,W){const V=W.wireframe===!0;let G=i[F.id];G===void 0&&(G={},i[F.id]=G);const z=I.isInstancedMesh===!0?I.id:0;let j=G[z];j===void 0&&(j={},G[z]=j);let $=j[B.id];$===void 0&&($={},j[B.id]=$);let Q=$[V];return Q===void 0&&(Q=f(c()),$[V]=Q),Q}function f(I){const F=[],B=[],W=[];for(let V=0;V<n;V++)F[V]=0,B[V]=0,W[V]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:F,enabledAttributes:B,attributeDivisors:W,object:I,attributes:{},index:null}}function g(I,F,B,W){const V=s.attributes,G=F.attributes;let z=0;const j=B.getAttributes();for(const $ in j)if(j[$].location>=0){const se=V[$];let ae=G[$];if(ae===void 0&&($==="instanceMatrix"&&I.instanceMatrix&&(ae=I.instanceMatrix),$==="instanceColor"&&I.instanceColor&&(ae=I.instanceColor)),se===void 0||se.attribute!==ae||ae&&se.data!==ae.data)return!0;z++}return s.attributesNum!==z||s.index!==W}function x(I,F,B,W){const V={},G=F.attributes;let z=0;const j=B.getAttributes();for(const $ in j)if(j[$].location>=0){let se=G[$];se===void 0&&($==="instanceMatrix"&&I.instanceMatrix&&(se=I.instanceMatrix),$==="instanceColor"&&I.instanceColor&&(se=I.instanceColor));const ae={};ae.attribute=se,se&&se.data&&(ae.data=se.data),V[$]=ae,z++}s.attributes=V,s.attributesNum=z,s.index=W}function M(){const I=s.newAttributes;for(let F=0,B=I.length;F<B;F++)I[F]=0}function v(I){d(I,0)}function d(I,F){const B=s.newAttributes,W=s.enabledAttributes,V=s.attributeDivisors;B[I]=1,W[I]===0&&(t.enableVertexAttribArray(I),W[I]=1),V[I]!==F&&(t.vertexAttribDivisor(I,F),V[I]=F)}function m(){const I=s.newAttributes,F=s.enabledAttributes;for(let B=0,W=F.length;B<W;B++)F[B]!==I[B]&&(t.disableVertexAttribArray(B),F[B]=0)}function _(I,F,B,W,V,G,z){z===!0?t.vertexAttribIPointer(I,F,B,V,G):t.vertexAttribPointer(I,F,B,W,V,G)}function b(I,F,B,W){M();const V=W.attributes,G=B.getAttributes(),z=F.defaultAttributeValues;for(const j in G){const $=G[j];if($.location>=0){let Q=V[j];if(Q===void 0&&(j==="instanceMatrix"&&I.instanceMatrix&&(Q=I.instanceMatrix),j==="instanceColor"&&I.instanceColor&&(Q=I.instanceColor)),Q!==void 0){const se=Q.normalized,ae=Q.itemSize,Ae=e.get(Q);if(Ae===void 0)continue;const De=Ae.buffer,Oe=Ae.type,D=Ae.bytesPerElement,q=Oe===t.INT||Oe===t.UNSIGNED_INT||Q.gpuType===Up;if(Q.isInterleavedBufferAttribute){const ne=Q.data,oe=ne.stride,ye=Q.offset;if(ne.isInstancedInterleavedBuffer){for(let Le=0;Le<$.locationSize;Le++)d($.location+Le,ne.meshPerAttribute);I.isInstancedMesh!==!0&&W._maxInstanceCount===void 0&&(W._maxInstanceCount=ne.meshPerAttribute*ne.count)}else for(let Le=0;Le<$.locationSize;Le++)v($.location+Le);t.bindBuffer(t.ARRAY_BUFFER,De);for(let Le=0;Le<$.locationSize;Le++)_($.location+Le,ae/$.locationSize,Oe,se,oe*D,(ye+ae/$.locationSize*Le)*D,q)}else{if(Q.isInstancedBufferAttribute){for(let ne=0;ne<$.locationSize;ne++)d($.location+ne,Q.meshPerAttribute);I.isInstancedMesh!==!0&&W._maxInstanceCount===void 0&&(W._maxInstanceCount=Q.meshPerAttribute*Q.count)}else for(let ne=0;ne<$.locationSize;ne++)v($.location+ne);t.bindBuffer(t.ARRAY_BUFFER,De);for(let ne=0;ne<$.locationSize;ne++)_($.location+ne,ae/$.locationSize,Oe,se,ae*D,ae/$.locationSize*ne*D,q)}}else if(z!==void 0){const se=z[j];if(se!==void 0)switch(se.length){case 2:t.vertexAttrib2fv($.location,se);break;case 3:t.vertexAttrib3fv($.location,se);break;case 4:t.vertexAttrib4fv($.location,se);break;default:t.vertexAttrib1fv($.location,se)}}}}m()}function w(){C();for(const I in i){const F=i[I];for(const B in F){const W=F[B];for(const V in W){const G=W[V];for(const z in G)p(G[z].object),delete G[z];delete W[V]}}delete i[I]}}function A(I){if(i[I.id]===void 0)return;const F=i[I.id];for(const B in F){const W=F[B];for(const V in W){const G=W[V];for(const z in G)p(G[z].object),delete G[z];delete W[V]}}delete i[I.id]}function E(I){for(const F in i){const B=i[F];for(const W in B){const V=B[W];if(V[I.id]===void 0)continue;const G=V[I.id];for(const z in G)p(G[z].object),delete G[z];delete V[I.id]}}}function y(I){for(const F in i){const B=i[F],W=I.isInstancedMesh===!0?I.id:0,V=B[W];if(V!==void 0){for(const G in V){const z=V[G];for(const j in z)p(z[j].object),delete z[j];delete V[G]}delete B[W],Object.keys(B).length===0&&delete i[F]}}}function C(){P(),o=!0,s!==r&&(s=r,u(s.object))}function P(){r.geometry=null,r.program=null,r.wireframe=!1}return{setup:a,reset:C,resetDefaultState:P,dispose:w,releaseStatesOfGeometry:A,releaseStatesOfObject:y,releaseStatesOfProgram:E,initAttributes:M,enableAttribute:v,disableUnusedAttributes:m}}function C2(t,e,n){let i;function r(u){i=u}function s(u,p){t.drawArrays(i,u,p),n.update(p,i,1)}function o(u,p,h){h!==0&&(t.drawArraysInstanced(i,u,p,h),n.update(p,i,h))}function a(u,p,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,u,0,p,0,h);let g=0;for(let x=0;x<h;x++)g+=p[x];n.update(g,i,1)}function c(u,p,h,f){if(h===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let x=0;x<u.length;x++)o(u[x],p[x],f[x]);else{g.multiDrawArraysInstancedWEBGL(i,u,0,p,0,f,0,h);let x=0;for(let M=0;M<h;M++)x+=p[M]*f[M];n.update(x,i,1)}}this.setMode=r,this.render=s,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=c}function A2(t,e,n,i){let r;function s(){if(r!==void 0)return r;if(e.has("EXT_texture_filter_anisotropic")===!0){const E=e.get("EXT_texture_filter_anisotropic");r=t.getParameter(E.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else r=0;return r}function o(E){return!(E!==Xn&&i.convert(E)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(E){const y=E===Ni&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(E!==bn&&i.convert(E)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_TYPE)&&E!==oi&&!y)}function c(E){if(E==="highp"){if(t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.HIGH_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.HIGH_FLOAT).precision>0)return"highp";E="mediump"}return E==="mediump"&&t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.MEDIUM_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let u=n.precision!==void 0?n.precision:"highp";const p=c(u);p!==u&&(He("WebGLRenderer:",u,"not supported, using",p,"instead."),u=p);const h=n.logarithmicDepthBuffer===!0,f=n.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),g=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS),x=t.getParameter(t.MAX_VERTEX_TEXTURE_IMAGE_UNITS),M=t.getParameter(t.MAX_TEXTURE_SIZE),v=t.getParameter(t.MAX_CUBE_MAP_TEXTURE_SIZE),d=t.getParameter(t.MAX_VERTEX_ATTRIBS),m=t.getParameter(t.MAX_VERTEX_UNIFORM_VECTORS),_=t.getParameter(t.MAX_VARYING_VECTORS),b=t.getParameter(t.MAX_FRAGMENT_UNIFORM_VECTORS),w=t.getParameter(t.MAX_SAMPLES),A=t.getParameter(t.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:s,getMaxPrecision:c,textureFormatReadable:o,textureTypeReadable:a,precision:u,logarithmicDepthBuffer:h,reversedDepthBuffer:f,maxTextures:g,maxVertexTextures:x,maxTextureSize:M,maxCubemapSize:v,maxAttributes:d,maxVertexUniforms:m,maxVaryings:_,maxFragmentUniforms:b,maxSamples:w,samples:A}}function R2(t){const e=this;let n=null,i=0,r=!1,s=!1;const o=new Ir,a=new qe,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(h,f){const g=h.length!==0||f||i!==0||r;return r=f,i=h.length,g},this.beginShadows=function(){s=!0,p(null)},this.endShadows=function(){s=!1},this.setGlobalState=function(h,f){n=p(h,f,0)},this.setState=function(h,f,g){const x=h.clippingPlanes,M=h.clipIntersection,v=h.clipShadows,d=t.get(h);if(!r||x===null||x.length===0||s&&!v)s?p(null):u();else{const m=s?0:i,_=m*4;let b=d.clippingState||null;c.value=b,b=p(x,f,_,g);for(let w=0;w!==_;++w)b[w]=n[w];d.clippingState=b,this.numIntersection=M?this.numPlanes:0,this.numPlanes+=m}};function u(){c.value!==n&&(c.value=n,c.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function p(h,f,g,x){const M=h!==null?h.length:0;let v=null;if(M!==0){if(v=c.value,x!==!0||v===null){const d=g+M*4,m=f.matrixWorldInverse;a.getNormalMatrix(m),(v===null||v.length<d)&&(v=new Float32Array(d));for(let _=0,b=g;_!==M;++_,b+=4)o.copy(h[_]).applyMatrix4(m,a),o.normal.toArray(v,b),v[b+3]=o.constant}c.value=v,c.needsUpdate=!0}return e.numPlanes=M,e.numIntersection=0,v}}const er=4,Ym=[.125,.215,.35,.446,.526,.582],Dr=20,I2=256,yo=new mx,Zm=new tt;let ju=null,Vu=0,Hu=0,Gu=!1;const P2=new H;class Jm{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,n=0,i=.1,r=100,s={}){const{size:o=256,position:a=P2}=s;ju=this._renderer.getRenderTarget(),Vu=this._renderer.getActiveCubeFace(),Hu=this._renderer.getActiveMipmapLevel(),Gu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const c=this._allocateTargets();return c.depthBuffer=!0,this._sceneToCubeUV(e,i,r,c,a),n>0&&this._blur(c,0,0,n),this._applyPMREM(c),this._cleanup(c),c}fromEquirectangular(e,n=null){return this._fromTexture(e,n)}fromCubemap(e,n=null){return this._fromTexture(e,n)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=t0(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=e0(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(ju,Vu,Hu),this._renderer.xr.enabled=Gu,e.scissorTest=!1,ms(e,0,0,e.width,e.height)}_fromTexture(e,n){e.mapping===Xr||e.mapping===Xs?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),ju=this._renderer.getRenderTarget(),Vu=this._renderer.getActiveCubeFace(),Hu=this._renderer.getActiveMipmapLevel(),Gu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const i=n||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),n=4*this._cubeSize,i={magFilter:Qt,minFilter:Qt,generateMipmaps:!1,type:Ni,format:Xn,colorSpace:$s,depthBuffer:!1},r=Qm(e,n,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==n){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Qm(e,n,i);const{_lodMax:s}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=D2(s)),this._blurMaterial=F2(s,e,n),this._ggxMaterial=L2(s,e,n)}return r}_compileMaterial(e){const n=new pn(new rn,e);this._renderer.compile(n,yo)}_sceneToCubeUV(e,n,i,r,s){const c=new Mn(90,1,n,i),u=[1,-1,1,1,1,1],p=[1,1,1,-1,-1,-1],h=this._renderer,f=h.autoClear,g=h.toneMapping;h.getClearColor(Zm),h.toneMapping=ui,h.autoClear=!1,h.state.buffers.depth.getReversed()&&(h.setRenderTarget(r),h.clearDepth(),h.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new pn(new ha,new As({name:"PMREM.Background",side:en,depthWrite:!1,depthTest:!1})));const M=this._backgroundBox,v=M.material;let d=!1;const m=e.background;m?m.isColor&&(v.color.copy(m),e.background=null,d=!0):(v.color.copy(Zm),d=!0);for(let _=0;_<6;_++){const b=_%3;b===0?(c.up.set(0,u[_],0),c.position.set(s.x,s.y,s.z),c.lookAt(s.x+p[_],s.y,s.z)):b===1?(c.up.set(0,0,u[_]),c.position.set(s.x,s.y,s.z),c.lookAt(s.x,s.y+p[_],s.z)):(c.up.set(0,u[_],0),c.position.set(s.x,s.y,s.z),c.lookAt(s.x,s.y,s.z+p[_]));const w=this._cubeSize;ms(r,b*w,_>2?w:0,w,w),h.setRenderTarget(r),d&&h.render(M,c),h.render(e,c)}h.toneMapping=g,h.autoClear=f,e.background=m}_textureToCubeUV(e,n){const i=this._renderer,r=e.mapping===Xr||e.mapping===Xs;r?(this._cubemapMaterial===null&&(this._cubemapMaterial=t0()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=e0());const s=r?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=s;const a=s.uniforms;a.envMap.value=e;const c=this._cubeSize;ms(n,0,0,3*c,2*c),i.setRenderTarget(n),i.render(o,yo)}_applyPMREM(e){const n=this._renderer,i=n.autoClear;n.autoClear=!1;const r=this._lodMeshes.length;for(let s=1;s<r;s++)this._applyGGXFilter(e,s-1,s);n.autoClear=i}_applyGGXFilter(e,n,i){const r=this._renderer,s=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[i];a.material=o;const c=o.uniforms,u=i/(this._lodMeshes.length-1),p=n/(this._lodMeshes.length-1),h=Math.sqrt(u*u-p*p),f=0+u*1.25,g=h*f,{_lodMax:x}=this,M=this._sizeLods[i],v=3*M*(i>x-er?i-x+er:0),d=4*(this._cubeSize-M);c.envMap.value=e.texture,c.roughness.value=g,c.mipInt.value=x-n,ms(s,v,d,3*M,2*M),r.setRenderTarget(s),r.render(a,yo),c.envMap.value=s.texture,c.roughness.value=0,c.mipInt.value=x-i,ms(e,v,d,3*M,2*M),r.setRenderTarget(e),r.render(a,yo)}_blur(e,n,i,r,s){const o=this._pingPongRenderTarget;this._halfBlur(e,o,n,i,r,"latitudinal",s),this._halfBlur(o,e,i,i,r,"longitudinal",s)}_halfBlur(e,n,i,r,s,o,a){const c=this._renderer,u=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&it("blur direction must be either latitudinal or longitudinal!");const p=3,h=this._lodMeshes[r];h.material=u;const f=u.uniforms,g=this._sizeLods[i]-1,x=isFinite(s)?Math.PI/(2*g):2*Math.PI/(2*Dr-1),M=s/x,v=isFinite(s)?1+Math.floor(p*M):Dr;v>Dr&&He(`sigmaRadians, ${s}, is too large and will clip, as it requested ${v} samples when the maximum is set to ${Dr}`);const d=[];let m=0;for(let E=0;E<Dr;++E){const y=E/M,C=Math.exp(-y*y/2);d.push(C),E===0?m+=C:E<v&&(m+=2*C)}for(let E=0;E<d.length;E++)d[E]=d[E]/m;f.envMap.value=e.texture,f.samples.value=v,f.weights.value=d,f.latitudinal.value=o==="latitudinal",a&&(f.poleAxis.value=a);const{_lodMax:_}=this;f.dTheta.value=x,f.mipInt.value=_-i;const b=this._sizeLods[r],w=3*b*(r>_-er?r-_+er:0),A=4*(this._cubeSize-b);ms(n,w,A,3*b,2*b),c.setRenderTarget(n),c.render(h,yo)}}function D2(t){const e=[],n=[],i=[];let r=t;const s=t-er+1+Ym.length;for(let o=0;o<s;o++){const a=Math.pow(2,r);e.push(a);let c=1/a;o>t-er?c=Ym[o-t+er-1]:o===0&&(c=0),n.push(c);const u=1/(a-2),p=-u,h=1+u,f=[p,p,h,p,h,h,p,p,h,h,p,h],g=6,x=6,M=3,v=2,d=1,m=new Float32Array(M*x*g),_=new Float32Array(v*x*g),b=new Float32Array(d*x*g);for(let A=0;A<g;A++){const E=A%3*2/3-1,y=A>2?0:-1,C=[E,y,0,E+2/3,y,0,E+2/3,y+1,0,E,y,0,E+2/3,y+1,0,E,y+1,0];m.set(C,M*x*A),_.set(f,v*x*A);const P=[A,A,A,A,A,A];b.set(P,d*x*A)}const w=new rn;w.setAttribute("position",new Kn(m,M)),w.setAttribute("uv",new Kn(_,v)),w.setAttribute("faceIndex",new Kn(b,d)),i.push(new pn(w,null)),r>er&&r--}return{lodMeshes:i,sizeLods:e,sigmas:n}}function Qm(t,e,n){const i=new di(t,e,n);return i.texture.mapping=Rc,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function ms(t,e,n,i,r){t.viewport.set(e,n,i,r),t.scissor.set(e,n,i,r)}function L2(t,e,n){return new hi({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:I2,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Pc(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:Ri,depthTest:!1,depthWrite:!1})}function F2(t,e,n){const i=new Float32Array(Dr),r=new H(0,1,0);return new hi({name:"SphericalGaussianBlur",defines:{n:Dr,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:r}},vertexShader:Pc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Ri,depthTest:!1,depthWrite:!1})}function e0(){return new hi({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Pc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Ri,depthTest:!1,depthWrite:!1})}function t0(){return new hi({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Pc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Ri,depthTest:!1,depthWrite:!1})}function Pc(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}class xx extends di{constructor(e=1,n={}){super(e,e,n),this.isWebGLCubeRenderTarget=!0;const i={width:e,height:e,depth:1},r=[i,i,i,i,i,i];this.texture=new ux(r),this._setTextureOptions(n),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,n){this.texture.type=n.type,this.texture.colorSpace=n.colorSpace,this.texture.generateMipmaps=n.generateMipmaps,this.texture.minFilter=n.minFilter,this.texture.magFilter=n.magFilter;const i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},r=new ha(5,5,5),s=new hi({name:"CubemapFromEquirect",uniforms:Ks(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:en,blending:Ri});s.uniforms.tEquirect.value=n;const o=new pn(r,s),a=n.minFilter;return n.minFilter===Ur&&(n.minFilter=Qt),new BS(1,10,this).update(e,o),n.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,n=!0,i=!0,r=!0){const s=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(n,i,r);e.setRenderTarget(s)}}function N2(t){let e=new WeakMap,n=new WeakMap,i=null;function r(f,g=!1){return f==null?null:g?o(f):s(f)}function s(f){if(f&&f.isTexture){const g=f.mapping;if(g===du||g===fu)if(e.has(f)){const x=e.get(f).texture;return a(x,f.mapping)}else{const x=f.image;if(x&&x.height>0){const M=new xx(x.height);return M.fromEquirectangularTexture(t,f),e.set(f,M),f.addEventListener("dispose",u),a(M.texture,f.mapping)}else return null}}return f}function o(f){if(f&&f.isTexture){const g=f.mapping,x=g===du||g===fu,M=g===Xr||g===Xs;if(x||M){let v=n.get(f);const d=v!==void 0?v.texture.pmremVersion:0;if(f.isRenderTargetTexture&&f.pmremVersion!==d)return i===null&&(i=new Jm(t)),v=x?i.fromEquirectangular(f,v):i.fromCubemap(f,v),v.texture.pmremVersion=f.pmremVersion,n.set(f,v),v.texture;if(v!==void 0)return v.texture;{const m=f.image;return x&&m&&m.height>0||M&&m&&c(m)?(i===null&&(i=new Jm(t)),v=x?i.fromEquirectangular(f):i.fromCubemap(f),v.texture.pmremVersion=f.pmremVersion,n.set(f,v),f.addEventListener("dispose",p),v.texture):null}}}return f}function a(f,g){return g===du?f.mapping=Xr:g===fu&&(f.mapping=Xs),f}function c(f){let g=0;const x=6;for(let M=0;M<x;M++)f[M]!==void 0&&g++;return g===x}function u(f){const g=f.target;g.removeEventListener("dispose",u);const x=e.get(g);x!==void 0&&(e.delete(g),x.dispose())}function p(f){const g=f.target;g.removeEventListener("dispose",p);const x=n.get(g);x!==void 0&&(n.delete(g),x.dispose())}function h(){e=new WeakMap,n=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:r,dispose:h}}function U2(t){const e={};function n(i){if(e[i]!==void 0)return e[i];const r=t.getExtension(i);return e[i]=r,r}return{has:function(i){return n(i)!==null},init:function(){n("EXT_color_buffer_float"),n("WEBGL_clip_cull_distance"),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float"),n("WEBGL_multisampled_render_to_texture"),n("WEBGL_render_shared_exponent")},get:function(i){const r=n(i);return r===null&&ac("WebGLRenderer: "+i+" extension not supported."),r}}}function k2(t,e,n,i){const r={},s=new WeakMap;function o(h){const f=h.target;f.index!==null&&e.remove(f.index);for(const x in f.attributes)e.remove(f.attributes[x]);f.removeEventListener("dispose",o),delete r[f.id];const g=s.get(f);g&&(e.remove(g),s.delete(f)),i.releaseStatesOfGeometry(f),f.isInstancedBufferGeometry===!0&&delete f._maxInstanceCount,n.memory.geometries--}function a(h,f){return r[f.id]===!0||(f.addEventListener("dispose",o),r[f.id]=!0,n.memory.geometries++),f}function c(h){const f=h.attributes;for(const g in f)e.update(f[g],t.ARRAY_BUFFER)}function u(h){const f=[],g=h.index,x=h.attributes.position;let M=0;if(x===void 0)return;if(g!==null){const m=g.array;M=g.version;for(let _=0,b=m.length;_<b;_+=3){const w=m[_+0],A=m[_+1],E=m[_+2];f.push(w,A,A,E,E,w)}}else{const m=x.array;M=x.version;for(let _=0,b=m.length/3-1;_<b;_+=3){const w=_+0,A=_+1,E=_+2;f.push(w,A,A,E,E,w)}}const v=new(x.count>=65535?sx:rx)(f,1);v.version=M;const d=s.get(h);d&&e.remove(d),s.set(h,v)}function p(h){const f=s.get(h);if(f){const g=h.index;g!==null&&f.version<g.version&&u(h)}else u(h);return s.get(h)}return{get:a,update:c,getWireframeAttribute:p}}function O2(t,e,n){let i;function r(f){i=f}let s,o;function a(f){s=f.type,o=f.bytesPerElement}function c(f,g){t.drawElements(i,g,s,f*o),n.update(g,i,1)}function u(f,g,x){x!==0&&(t.drawElementsInstanced(i,g,s,f*o,x),n.update(g,i,x))}function p(f,g,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,g,0,s,f,0,x);let v=0;for(let d=0;d<x;d++)v+=g[d];n.update(v,i,1)}function h(f,g,x,M){if(x===0)return;const v=e.get("WEBGL_multi_draw");if(v===null)for(let d=0;d<f.length;d++)u(f[d]/o,g[d],M[d]);else{v.multiDrawElementsInstancedWEBGL(i,g,0,s,f,0,M,0,x);let d=0;for(let m=0;m<x;m++)d+=g[m]*M[m];n.update(d,i,1)}}this.setMode=r,this.setIndex=a,this.render=c,this.renderInstances=u,this.renderMultiDraw=p,this.renderMultiDrawInstances=h}function z2(t){const e={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function i(s,o,a){switch(n.calls++,o){case t.TRIANGLES:n.triangles+=a*(s/3);break;case t.LINES:n.lines+=a*(s/2);break;case t.LINE_STRIP:n.lines+=a*(s-1);break;case t.LINE_LOOP:n.lines+=a*s;break;case t.POINTS:n.points+=a*s;break;default:it("WebGLInfo: Unknown draw mode:",o);break}}function r(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:e,render:n,programs:null,autoReset:!0,reset:r,update:i}}function B2(t,e,n){const i=new WeakMap,r=new wt;function s(o,a,c){const u=o.morphTargetInfluences,p=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,h=p!==void 0?p.length:0;let f=i.get(a);if(f===void 0||f.count!==h){let P=function(){y.dispose(),i.delete(a),a.removeEventListener("dispose",P)};var g=P;f!==void 0&&f.texture.dispose();const x=a.morphAttributes.position!==void 0,M=a.morphAttributes.normal!==void 0,v=a.morphAttributes.color!==void 0,d=a.morphAttributes.position||[],m=a.morphAttributes.normal||[],_=a.morphAttributes.color||[];let b=0;x===!0&&(b=1),M===!0&&(b=2),v===!0&&(b=3);let w=a.attributes.position.count*b,A=1;w>e.maxTextureSize&&(A=Math.ceil(w/e.maxTextureSize),w=e.maxTextureSize);const E=new Float32Array(w*A*4*h),y=new tx(E,w,A,h);y.type=oi,y.needsUpdate=!0;const C=b*4;for(let I=0;I<h;I++){const F=d[I],B=m[I],W=_[I],V=w*A*4*I;for(let G=0;G<F.count;G++){const z=G*C;x===!0&&(r.fromBufferAttribute(F,G),E[V+z+0]=r.x,E[V+z+1]=r.y,E[V+z+2]=r.z,E[V+z+3]=0),M===!0&&(r.fromBufferAttribute(B,G),E[V+z+4]=r.x,E[V+z+5]=r.y,E[V+z+6]=r.z,E[V+z+7]=0),v===!0&&(r.fromBufferAttribute(W,G),E[V+z+8]=r.x,E[V+z+9]=r.y,E[V+z+10]=r.z,E[V+z+11]=W.itemSize===4?r.w:1)}}f={count:h,texture:y,size:new Ze(w,A)},i.set(a,f),a.addEventListener("dispose",P)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)c.getUniforms().setValue(t,"morphTexture",o.morphTexture,n);else{let x=0;for(let v=0;v<u.length;v++)x+=u[v];const M=a.morphTargetsRelative?1:1-x;c.getUniforms().setValue(t,"morphTargetBaseInfluence",M),c.getUniforms().setValue(t,"morphTargetInfluences",u)}c.getUniforms().setValue(t,"morphTargetsTexture",f.texture,n),c.getUniforms().setValue(t,"morphTargetsTextureSize",f.size)}return{update:s}}function j2(t,e,n,i,r){let s=new WeakMap;function o(u){const p=r.render.frame,h=u.geometry,f=e.get(u,h);if(s.get(f)!==p&&(e.update(f),s.set(f,p)),u.isInstancedMesh&&(u.hasEventListener("dispose",c)===!1&&u.addEventListener("dispose",c),s.get(u)!==p&&(n.update(u.instanceMatrix,t.ARRAY_BUFFER),u.instanceColor!==null&&n.update(u.instanceColor,t.ARRAY_BUFFER),s.set(u,p))),u.isSkinnedMesh){const g=u.skeleton;s.get(g)!==p&&(g.update(),s.set(g,p))}return f}function a(){s=new WeakMap}function c(u){const p=u.target;p.removeEventListener("dispose",c),i.releaseStatesOfObject(p),n.remove(p.instanceMatrix),p.instanceColor!==null&&n.remove(p.instanceColor)}return{update:o,dispose:a}}const V2={[O1]:"LINEAR_TONE_MAPPING",[z1]:"REINHARD_TONE_MAPPING",[B1]:"CINEON_TONE_MAPPING",[j1]:"ACES_FILMIC_TONE_MAPPING",[H1]:"AGX_TONE_MAPPING",[G1]:"NEUTRAL_TONE_MAPPING",[V1]:"CUSTOM_TONE_MAPPING"};function H2(t,e,n,i,r){const s=new di(e,n,{type:t,depthBuffer:i,stencilBuffer:r}),o=new di(e,n,{type:Ni,depthBuffer:!1,stencilBuffer:!1}),a=new rn;a.setAttribute("position",new Wt([-1,3,0,-1,-1,0,3,-1,0],3)),a.setAttribute("uv",new Wt([0,2,0,0,2,0],2));const c=new DS({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),u=new pn(a,c),p=new mx(-1,1,1,-1,0,1);let h=null,f=null,g=!1,x,M=null,v=[],d=!1;this.setSize=function(m,_){s.setSize(m,_),o.setSize(m,_);for(let b=0;b<v.length;b++){const w=v[b];w.setSize&&w.setSize(m,_)}},this.setEffects=function(m){v=m,d=v.length>0&&v[0].isRenderPass===!0;const _=s.width,b=s.height;for(let w=0;w<v.length;w++){const A=v[w];A.setSize&&A.setSize(_,b)}},this.begin=function(m,_){if(g||m.toneMapping===ui&&v.length===0)return!1;if(M=_,_!==null){const b=_.width,w=_.height;(s.width!==b||s.height!==w)&&this.setSize(b,w)}return d===!1&&m.setRenderTarget(s),x=m.toneMapping,m.toneMapping=ui,!0},this.hasRenderPass=function(){return d},this.end=function(m,_){m.toneMapping=x,g=!0;let b=s,w=o;for(let A=0;A<v.length;A++){const E=v[A];if(E.enabled!==!1&&(E.render(m,w,b,_),E.needsSwap!==!1)){const y=b;b=w,w=y}}if(h!==m.outputColorSpace||f!==m.toneMapping){h=m.outputColorSpace,f=m.toneMapping,c.defines={},rt.getTransfer(h)===lt&&(c.defines.SRGB_TRANSFER="");const A=V2[f];A&&(c.defines[A]=""),c.needsUpdate=!0}c.uniforms.tDiffuse.value=b.texture,m.setRenderTarget(M),m.render(u,p),M=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){s.dispose(),o.dispose(),a.dispose(),c.dispose()}}const vx=new tn,Ff=new ia(1,1),_x=new tx,yx=new rS,Sx=new ux,n0=[],i0=[],r0=new Float32Array(16),s0=new Float32Array(9),o0=new Float32Array(4);function no(t,e,n){const i=t[0];if(i<=0||i>0)return t;const r=e*n;let s=n0[r];if(s===void 0&&(s=new Float32Array(r),n0[r]=s),e!==0){i.toArray(s,0);for(let o=1,a=0;o!==e;++o)a+=n,t[o].toArray(s,a)}return s}function Lt(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function Ft(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function Dc(t,e){let n=i0[e];n===void 0&&(n=new Int32Array(e),i0[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function G2(t,e){const n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function W2(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(Lt(n,e))return;t.uniform2fv(this.addr,e),Ft(n,e)}}function X2(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(e.r!==void 0)(n[0]!==e.r||n[1]!==e.g||n[2]!==e.b)&&(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(Lt(n,e))return;t.uniform3fv(this.addr,e),Ft(n,e)}}function q2(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(Lt(n,e))return;t.uniform4fv(this.addr,e),Ft(n,e)}}function $2(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(Lt(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),Ft(n,e)}else{if(Lt(n,i))return;o0.set(i),t.uniformMatrix2fv(this.addr,!1,o0),Ft(n,i)}}function K2(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(Lt(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),Ft(n,e)}else{if(Lt(n,i))return;s0.set(i),t.uniformMatrix3fv(this.addr,!1,s0),Ft(n,i)}}function Y2(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(Lt(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),Ft(n,e)}else{if(Lt(n,i))return;r0.set(i),t.uniformMatrix4fv(this.addr,!1,r0),Ft(n,i)}}function Z2(t,e){const n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function J2(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2i(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(Lt(n,e))return;t.uniform2iv(this.addr,e),Ft(n,e)}}function Q2(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3i(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(Lt(n,e))return;t.uniform3iv(this.addr,e),Ft(n,e)}}function eE(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4i(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(Lt(n,e))return;t.uniform4iv(this.addr,e),Ft(n,e)}}function tE(t,e){const n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function nE(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2ui(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(Lt(n,e))return;t.uniform2uiv(this.addr,e),Ft(n,e)}}function iE(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3ui(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(Lt(n,e))return;t.uniform3uiv(this.addr,e),Ft(n,e)}}function rE(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4ui(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(Lt(n,e))return;t.uniform4uiv(this.addr,e),Ft(n,e)}}function sE(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r);let s;this.type===t.SAMPLER_2D_SHADOW?(Ff.compareFunction=n.isReversedDepthBuffer()?Hp:Vp,s=Ff):s=vx,n.setTexture2D(e||s,r)}function oE(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture3D(e||yx,r)}function aE(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTextureCube(e||Sx,r)}function lE(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture2DArray(e||_x,r)}function cE(t){switch(t){case 5126:return G2;case 35664:return W2;case 35665:return X2;case 35666:return q2;case 35674:return $2;case 35675:return K2;case 35676:return Y2;case 5124:case 35670:return Z2;case 35667:case 35671:return J2;case 35668:case 35672:return Q2;case 35669:case 35673:return eE;case 5125:return tE;case 36294:return nE;case 36295:return iE;case 36296:return rE;case 35678:case 36198:case 36298:case 36306:case 35682:return sE;case 35679:case 36299:case 36307:return oE;case 35680:case 36300:case 36308:case 36293:return aE;case 36289:case 36303:case 36311:case 36292:return lE}}function uE(t,e){t.uniform1fv(this.addr,e)}function dE(t,e){const n=no(e,this.size,2);t.uniform2fv(this.addr,n)}function fE(t,e){const n=no(e,this.size,3);t.uniform3fv(this.addr,n)}function pE(t,e){const n=no(e,this.size,4);t.uniform4fv(this.addr,n)}function hE(t,e){const n=no(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function mE(t,e){const n=no(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function gE(t,e){const n=no(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function xE(t,e){t.uniform1iv(this.addr,e)}function vE(t,e){t.uniform2iv(this.addr,e)}function _E(t,e){t.uniform3iv(this.addr,e)}function yE(t,e){t.uniform4iv(this.addr,e)}function SE(t,e){t.uniform1uiv(this.addr,e)}function ME(t,e){t.uniform2uiv(this.addr,e)}function bE(t,e){t.uniform3uiv(this.addr,e)}function EE(t,e){t.uniform4uiv(this.addr,e)}function TE(t,e,n){const i=this.cache,r=e.length,s=Dc(n,r);Lt(i,s)||(t.uniform1iv(this.addr,s),Ft(i,s));let o;this.type===t.SAMPLER_2D_SHADOW?o=Ff:o=vx;for(let a=0;a!==r;++a)n.setTexture2D(e[a]||o,s[a])}function wE(t,e,n){const i=this.cache,r=e.length,s=Dc(n,r);Lt(i,s)||(t.uniform1iv(this.addr,s),Ft(i,s));for(let o=0;o!==r;++o)n.setTexture3D(e[o]||yx,s[o])}function CE(t,e,n){const i=this.cache,r=e.length,s=Dc(n,r);Lt(i,s)||(t.uniform1iv(this.addr,s),Ft(i,s));for(let o=0;o!==r;++o)n.setTextureCube(e[o]||Sx,s[o])}function AE(t,e,n){const i=this.cache,r=e.length,s=Dc(n,r);Lt(i,s)||(t.uniform1iv(this.addr,s),Ft(i,s));for(let o=0;o!==r;++o)n.setTexture2DArray(e[o]||_x,s[o])}function RE(t){switch(t){case 5126:return uE;case 35664:return dE;case 35665:return fE;case 35666:return pE;case 35674:return hE;case 35675:return mE;case 35676:return gE;case 5124:case 35670:return xE;case 35667:case 35671:return vE;case 35668:case 35672:return _E;case 35669:case 35673:return yE;case 5125:return SE;case 36294:return ME;case 36295:return bE;case 36296:return EE;case 35678:case 36198:case 36298:case 36306:case 35682:return TE;case 35679:case 36299:case 36307:return wE;case 35680:case 36300:case 36308:case 36293:return CE;case 36289:case 36303:case 36311:case 36292:return AE}}class IE{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.setValue=cE(n.type)}}class PE{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.size=n.size,this.setValue=RE(n.type)}}class DE{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,n,i){const r=this.seq;for(let s=0,o=r.length;s!==o;++s){const a=r[s];a.setValue(e,n[a.id],i)}}}const Wu=/(\w+)(\])?(\[|\.)?/g;function a0(t,e){t.seq.push(e),t.map[e.id]=e}function LE(t,e,n){const i=t.name,r=i.length;for(Wu.lastIndex=0;;){const s=Wu.exec(i),o=Wu.lastIndex;let a=s[1];const c=s[2]==="]",u=s[3];if(c&&(a=a|0),u===void 0||u==="["&&o+2===r){a0(n,u===void 0?new IE(a,t,e):new PE(a,t,e));break}else{let h=n.map[a];h===void 0&&(h=new DE(a),a0(n,h)),n=h}}}class Pl{constructor(e,n){this.seq=[],this.map={};const i=e.getProgramParameter(n,e.ACTIVE_UNIFORMS);for(let o=0;o<i;++o){const a=e.getActiveUniform(n,o),c=e.getUniformLocation(n,a.name);LE(a,c,this)}const r=[],s=[];for(const o of this.seq)o.type===e.SAMPLER_2D_SHADOW||o.type===e.SAMPLER_CUBE_SHADOW||o.type===e.SAMPLER_2D_ARRAY_SHADOW?r.push(o):s.push(o);r.length>0&&(this.seq=r.concat(s))}setValue(e,n,i,r){const s=this.map[n];s!==void 0&&s.setValue(e,i,r)}setOptional(e,n,i){const r=n[i];r!==void 0&&this.setValue(e,i,r)}static upload(e,n,i,r){for(let s=0,o=n.length;s!==o;++s){const a=n[s],c=i[a.id];c.needsUpdate!==!1&&a.setValue(e,c.value,r)}}static seqWithValue(e,n){const i=[];for(let r=0,s=e.length;r!==s;++r){const o=e[r];o.id in n&&i.push(o)}return i}}function l0(t,e,n){const i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}const FE=37297;let NE=0;function UE(t,e){const n=t.split(`
`),i=[],r=Math.max(e-6,0),s=Math.min(e+6,n.length);for(let o=r;o<s;o++){const a=o+1;i.push(`${a===e?">":" "} ${a}: ${n[o]}`)}return i.join(`
`)}const c0=new qe;function kE(t){rt._getMatrix(c0,rt.workingColorSpace,t);const e=`mat3( ${c0.elements.map(n=>n.toFixed(4))} )`;switch(rt.getTransfer(t)){case rc:return[e,"LinearTransferOETF"];case lt:return[e,"sRGBTransferOETF"];default:return He("WebGLProgram: Unsupported color space: ",t),[e,"LinearTransferOETF"]}}function u0(t,e,n){const i=t.getShaderParameter(e,t.COMPILE_STATUS),s=(t.getShaderInfoLog(e)||"").trim();if(i&&s==="")return"";const o=/ERROR: 0:(\d+)/.exec(s);if(o){const a=parseInt(o[1]);return n.toUpperCase()+`

`+s+`

`+UE(t.getShaderSource(e),a)}else return s}function OE(t,e){const n=kE(e);return[`vec4 ${t}( vec4 value ) {`,`	return ${n[1]}( vec4( value.rgb * ${n[0]}, value.a ) );`,"}"].join(`
`)}const zE={[O1]:"Linear",[z1]:"Reinhard",[B1]:"Cineon",[j1]:"ACESFilmic",[H1]:"AgX",[G1]:"Neutral",[V1]:"Custom"};function BE(t,e){const n=zE[e];return n===void 0?(He("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+t+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}const fl=new H;function jE(){rt.getLuminanceCoefficients(fl);const t=fl.x.toFixed(4),e=fl.y.toFixed(4),n=fl.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${t}, ${e}, ${n} );`,"	return dot( weights, rgb );","}"].join(`
`)}function VE(t){return[t.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",t.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Co).join(`
`)}function HE(t){const e=[];for(const n in t){const i=t[n];i!==!1&&e.push("#define "+n+" "+i)}return e.join(`
`)}function GE(t,e){const n={},i=t.getProgramParameter(e,t.ACTIVE_ATTRIBUTES);for(let r=0;r<i;r++){const s=t.getActiveAttrib(e,r),o=s.name;let a=1;s.type===t.FLOAT_MAT2&&(a=2),s.type===t.FLOAT_MAT3&&(a=3),s.type===t.FLOAT_MAT4&&(a=4),n[o]={type:s.type,location:t.getAttribLocation(e,o),locationSize:a}}return n}function Co(t){return t!==""}function d0(t,e){const n=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return t.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function f0(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const WE=/^[ \t]*#include +<([\w\d./]+)>/gm;function Nf(t){return t.replace(WE,qE)}const XE=new Map;function qE(t,e){let n=Ke[e];if(n===void 0){const i=XE.get(e);if(i!==void 0)n=Ke[i],He('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("Can not resolve #include <"+e+">")}return Nf(n)}const $E=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function p0(t){return t.replace($E,KE)}function KE(t,e,n,i){let r="";for(let s=parseInt(e);s<parseInt(n);s++)r+=i.replace(/\[\s*i\s*\]/g,"[ "+s+" ]").replace(/UNROLLED_LOOP_INDEX/g,s);return r}function h0(t){let e=`precision ${t.precision} float;
	precision ${t.precision} int;
	precision ${t.precision} sampler2D;
	precision ${t.precision} samplerCube;
	precision ${t.precision} sampler3D;
	precision ${t.precision} sampler2DArray;
	precision ${t.precision} sampler2DShadow;
	precision ${t.precision} samplerCubeShadow;
	precision ${t.precision} sampler2DArrayShadow;
	precision ${t.precision} isampler2D;
	precision ${t.precision} isampler3D;
	precision ${t.precision} isamplerCube;
	precision ${t.precision} isampler2DArray;
	precision ${t.precision} usampler2D;
	precision ${t.precision} usampler3D;
	precision ${t.precision} usamplerCube;
	precision ${t.precision} usampler2DArray;
	`;return t.precision==="highp"?e+=`
#define HIGH_PRECISION`:t.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:t.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}const YE={[wl]:"SHADOWMAP_TYPE_PCF",[wo]:"SHADOWMAP_TYPE_VSM"};function ZE(t){return YE[t.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}const JE={[Xr]:"ENVMAP_TYPE_CUBE",[Xs]:"ENVMAP_TYPE_CUBE",[Rc]:"ENVMAP_TYPE_CUBE_UV"};function QE(t){return t.envMap===!1?"ENVMAP_TYPE_CUBE":JE[t.envMapMode]||"ENVMAP_TYPE_CUBE"}const eT={[Xs]:"ENVMAP_MODE_REFRACTION"};function tT(t){return t.envMap===!1?"ENVMAP_MODE_REFLECTION":eT[t.envMapMode]||"ENVMAP_MODE_REFLECTION"}const nT={[k1]:"ENVMAP_BLENDING_MULTIPLY",[ky]:"ENVMAP_BLENDING_MIX",[Oy]:"ENVMAP_BLENDING_ADD"};function iT(t){return t.envMap===!1?"ENVMAP_BLENDING_NONE":nT[t.combine]||"ENVMAP_BLENDING_NONE"}function rT(t){const e=t.envMapCubeUVHeight;if(e===null)return null;const n=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,n),7*16)),texelHeight:i,maxMip:n}}function sT(t,e,n,i){const r=t.getContext(),s=n.defines;let o=n.vertexShader,a=n.fragmentShader;const c=ZE(n),u=QE(n),p=tT(n),h=iT(n),f=rT(n),g=VE(n),x=HE(s),M=r.createProgram();let v,d,m=n.glslVersion?"#version "+n.glslVersion+`
`:"";n.isRawShaderMaterial?(v=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,x].filter(Co).join(`
`),v.length>0&&(v+=`
`),d=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,x].filter(Co).join(`
`),d.length>0&&(d+=`
`)):(v=[h0(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,x,n.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",n.batching?"#define USE_BATCHING":"",n.batchingColor?"#define USE_BATCHING_COLOR":"",n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.instancingMorph?"#define USE_INSTANCING_MORPH":"",n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+p:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.displacementMap?"#define USE_DISPLACEMENTMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.mapUv?"#define MAP_UV "+n.mapUv:"",n.alphaMapUv?"#define ALPHAMAP_UV "+n.alphaMapUv:"",n.lightMapUv?"#define LIGHTMAP_UV "+n.lightMapUv:"",n.aoMapUv?"#define AOMAP_UV "+n.aoMapUv:"",n.emissiveMapUv?"#define EMISSIVEMAP_UV "+n.emissiveMapUv:"",n.bumpMapUv?"#define BUMPMAP_UV "+n.bumpMapUv:"",n.normalMapUv?"#define NORMALMAP_UV "+n.normalMapUv:"",n.displacementMapUv?"#define DISPLACEMENTMAP_UV "+n.displacementMapUv:"",n.metalnessMapUv?"#define METALNESSMAP_UV "+n.metalnessMapUv:"",n.roughnessMapUv?"#define ROUGHNESSMAP_UV "+n.roughnessMapUv:"",n.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+n.anisotropyMapUv:"",n.clearcoatMapUv?"#define CLEARCOATMAP_UV "+n.clearcoatMapUv:"",n.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+n.clearcoatNormalMapUv:"",n.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+n.clearcoatRoughnessMapUv:"",n.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+n.iridescenceMapUv:"",n.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+n.iridescenceThicknessMapUv:"",n.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+n.sheenColorMapUv:"",n.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+n.sheenRoughnessMapUv:"",n.specularMapUv?"#define SPECULARMAP_UV "+n.specularMapUv:"",n.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+n.specularColorMapUv:"",n.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+n.specularIntensityMapUv:"",n.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+n.transmissionMapUv:"",n.thicknessMapUv?"#define THICKNESSMAP_UV "+n.thicknessMapUv:"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&n.flatShading===!1?"#define USE_MORPHNORMALS":"",n.morphColors?"#define USE_MORPHCOLORS":"",n.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+n.morphTextureStride:"",n.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+c:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Co).join(`
`),d=[h0(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,x,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+u:"",n.envMap?"#define "+p:"",n.envMap?"#define "+h:"",f?"#define CUBEUV_TEXEL_WIDTH "+f.texelWidth:"",f?"#define CUBEUV_TEXEL_HEIGHT "+f.texelHeight:"",f?"#define CUBEUV_MAX_MIP "+f.maxMip+".0":"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.dispersion?"#define USE_DISPERSION":"",n.iridescence?"#define USE_IRIDESCENCE":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas||n.batchingColor?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+c:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",n.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",n.toneMapping!==ui?"#define TONE_MAPPING":"",n.toneMapping!==ui?Ke.tonemapping_pars_fragment:"",n.toneMapping!==ui?BE("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.opaque?"#define OPAQUE":"",Ke.colorspace_pars_fragment,OE("linearToOutputTexel",n.outputColorSpace),jE(),n.useDepthPacking?"#define DEPTH_PACKING "+n.depthPacking:"",`
`].filter(Co).join(`
`)),o=Nf(o),o=d0(o,n),o=f0(o,n),a=Nf(a),a=d0(a,n),a=f0(a,n),o=p0(o),a=p0(a),n.isRawShaderMaterial!==!0&&(m=`#version 300 es
`,v=[g,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+v,d=["#define varying in",n.glslVersion===bm?"":"layout(location = 0) out highp vec4 pc_fragColor;",n.glslVersion===bm?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+d);const _=m+v+o,b=m+d+a,w=l0(r,r.VERTEX_SHADER,_),A=l0(r,r.FRAGMENT_SHADER,b);r.attachShader(M,w),r.attachShader(M,A),n.index0AttributeName!==void 0?r.bindAttribLocation(M,0,n.index0AttributeName):n.morphTargets===!0&&r.bindAttribLocation(M,0,"position"),r.linkProgram(M);function E(I){if(t.debug.checkShaderErrors){const F=r.getProgramInfoLog(M)||"",B=r.getShaderInfoLog(w)||"",W=r.getShaderInfoLog(A)||"",V=F.trim(),G=B.trim(),z=W.trim();let j=!0,$=!0;if(r.getProgramParameter(M,r.LINK_STATUS)===!1)if(j=!1,typeof t.debug.onShaderError=="function")t.debug.onShaderError(r,M,w,A);else{const Q=u0(r,w,"vertex"),se=u0(r,A,"fragment");it("THREE.WebGLProgram: Shader Error "+r.getError()+" - VALIDATE_STATUS "+r.getProgramParameter(M,r.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+V+`
`+Q+`
`+se)}else V!==""?He("WebGLProgram: Program Info Log:",V):(G===""||z==="")&&($=!1);$&&(I.diagnostics={runnable:j,programLog:V,vertexShader:{log:G,prefix:v},fragmentShader:{log:z,prefix:d}})}r.deleteShader(w),r.deleteShader(A),y=new Pl(r,M),C=GE(r,M)}let y;this.getUniforms=function(){return y===void 0&&E(this),y};let C;this.getAttributes=function(){return C===void 0&&E(this),C};let P=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return P===!1&&(P=r.getProgramParameter(M,FE)),P},this.destroy=function(){i.releaseStatesOfProgram(this),r.deleteProgram(M),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=NE++,this.cacheKey=e,this.usedTimes=1,this.program=M,this.vertexShader=w,this.fragmentShader=A,this}let oT=0;class aT{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const n=e.vertexShader,i=e.fragmentShader,r=this._getShaderStage(n),s=this._getShaderStage(i),o=this._getShaderCacheForMaterial(e);return o.has(r)===!1&&(o.add(r),r.usedTimes++),o.has(s)===!1&&(o.add(s),s.usedTimes++),this}remove(e){const n=this.materialCache.get(e);for(const i of n)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const n=this.materialCache;let i=n.get(e);return i===void 0&&(i=new Set,n.set(e,i)),i}_getShaderStage(e){const n=this.shaderCache;let i=n.get(e);return i===void 0&&(i=new lT(e),n.set(e,i)),i}}class lT{constructor(e){this.id=oT++,this.code=e,this.usedTimes=0}}function cT(t,e,n,i,r,s){const o=new nx,a=new aT,c=new Set,u=[],p=new Map,h=i.logarithmicDepthBuffer;let f=i.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function x(y){return c.add(y),y===0?"uv":`uv${y}`}function M(y,C,P,I,F){const B=I.fog,W=F.geometry,V=y.isMeshStandardMaterial||y.isMeshLambertMaterial||y.isMeshPhongMaterial?I.environment:null,G=y.isMeshStandardMaterial||y.isMeshLambertMaterial&&!y.envMap||y.isMeshPhongMaterial&&!y.envMap,z=e.get(y.envMap||V,G),j=z&&z.mapping===Rc?z.image.height:null,$=g[y.type];y.precision!==null&&(f=i.getMaxPrecision(y.precision),f!==y.precision&&He("WebGLProgram.getParameters:",y.precision,"not supported, using",f,"instead."));const Q=W.morphAttributes.position||W.morphAttributes.normal||W.morphAttributes.color,se=Q!==void 0?Q.length:0;let ae=0;W.morphAttributes.position!==void 0&&(ae=1),W.morphAttributes.normal!==void 0&&(ae=2),W.morphAttributes.color!==void 0&&(ae=3);let Ae,De,Oe,D;if($){const at=ni[$];Ae=at.vertexShader,De=at.fragmentShader}else Ae=y.vertexShader,De=y.fragmentShader,a.update(y),Oe=a.getVertexShaderID(y),D=a.getFragmentShaderID(y);const q=t.getRenderTarget(),ne=t.state.buffers.depth.getReversed(),oe=F.isInstancedMesh===!0,ye=F.isBatchedMesh===!0,Le=!!y.map,ht=!!y.matcap,ve=!!z,Ve=!!y.aoMap,ie=!!y.lightMap,le=!!y.bumpMap,ze=!!y.normalMap,L=!!y.displacementMap,Re=!!y.emissiveMap,Ge=!!y.metalnessMap,$e=!!y.roughnessMap,Te=y.anisotropy>0,R=y.clearcoat>0,S=y.dispersion>0,k=y.iridescence>0,te=y.sheen>0,re=y.transmission>0,J=Te&&!!y.anisotropyMap,be=R&&!!y.clearcoatMap,ge=R&&!!y.clearcoatNormalMap,Ue=R&&!!y.clearcoatRoughnessMap,Be=k&&!!y.iridescenceMap,de=k&&!!y.iridescenceThicknessMap,me=te&&!!y.sheenColorMap,N=te&&!!y.sheenRoughnessMap,ce=!!y.specularMap,ue=!!y.specularColorMap,je=!!y.specularIntensityMap,U=re&&!!y.transmissionMap,xe=re&&!!y.thicknessMap,pe=!!y.gradientMap,Ee=!!y.alphaMap,he=y.alphaTest>0,ee=!!y.alphaHash,Ie=!!y.extensions;let We=ui;y.toneMapped&&(q===null||q.isXRRenderTarget===!0)&&(We=t.toneMapping);const mt={shaderID:$,shaderType:y.type,shaderName:y.name,vertexShader:Ae,fragmentShader:De,defines:y.defines,customVertexShaderID:Oe,customFragmentShaderID:D,isRawShaderMaterial:y.isRawShaderMaterial===!0,glslVersion:y.glslVersion,precision:f,batching:ye,batchingColor:ye&&F._colorsTexture!==null,instancing:oe,instancingColor:oe&&F.instanceColor!==null,instancingMorph:oe&&F.morphTexture!==null,outputColorSpace:q===null?t.outputColorSpace:q.isXRRenderTarget===!0?q.texture.colorSpace:$s,alphaToCoverage:!!y.alphaToCoverage,map:Le,matcap:ht,envMap:ve,envMapMode:ve&&z.mapping,envMapCubeUVHeight:j,aoMap:Ve,lightMap:ie,bumpMap:le,normalMap:ze,displacementMap:L,emissiveMap:Re,normalMapObjectSpace:ze&&y.normalMapType===jy,normalMapTangentSpace:ze&&y.normalMapType===Q1,metalnessMap:Ge,roughnessMap:$e,anisotropy:Te,anisotropyMap:J,clearcoat:R,clearcoatMap:be,clearcoatNormalMap:ge,clearcoatRoughnessMap:Ue,dispersion:S,iridescence:k,iridescenceMap:Be,iridescenceThicknessMap:de,sheen:te,sheenColorMap:me,sheenRoughnessMap:N,specularMap:ce,specularColorMap:ue,specularIntensityMap:je,transmission:re,transmissionMap:U,thicknessMap:xe,gradientMap:pe,opaque:y.transparent===!1&&y.blending===Us&&y.alphaToCoverage===!1,alphaMap:Ee,alphaTest:he,alphaHash:ee,combine:y.combine,mapUv:Le&&x(y.map.channel),aoMapUv:Ve&&x(y.aoMap.channel),lightMapUv:ie&&x(y.lightMap.channel),bumpMapUv:le&&x(y.bumpMap.channel),normalMapUv:ze&&x(y.normalMap.channel),displacementMapUv:L&&x(y.displacementMap.channel),emissiveMapUv:Re&&x(y.emissiveMap.channel),metalnessMapUv:Ge&&x(y.metalnessMap.channel),roughnessMapUv:$e&&x(y.roughnessMap.channel),anisotropyMapUv:J&&x(y.anisotropyMap.channel),clearcoatMapUv:be&&x(y.clearcoatMap.channel),clearcoatNormalMapUv:ge&&x(y.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Ue&&x(y.clearcoatRoughnessMap.channel),iridescenceMapUv:Be&&x(y.iridescenceMap.channel),iridescenceThicknessMapUv:de&&x(y.iridescenceThicknessMap.channel),sheenColorMapUv:me&&x(y.sheenColorMap.channel),sheenRoughnessMapUv:N&&x(y.sheenRoughnessMap.channel),specularMapUv:ce&&x(y.specularMap.channel),specularColorMapUv:ue&&x(y.specularColorMap.channel),specularIntensityMapUv:je&&x(y.specularIntensityMap.channel),transmissionMapUv:U&&x(y.transmissionMap.channel),thicknessMapUv:xe&&x(y.thicknessMap.channel),alphaMapUv:Ee&&x(y.alphaMap.channel),vertexTangents:!!W.attributes.tangent&&(ze||Te),vertexColors:y.vertexColors,vertexAlphas:y.vertexColors===!0&&!!W.attributes.color&&W.attributes.color.itemSize===4,pointsUvs:F.isPoints===!0&&!!W.attributes.uv&&(Le||Ee),fog:!!B,useFog:y.fog===!0,fogExp2:!!B&&B.isFogExp2,flatShading:y.wireframe===!1&&(y.flatShading===!0||W.attributes.normal===void 0&&ze===!1&&(y.isMeshLambertMaterial||y.isMeshPhongMaterial||y.isMeshStandardMaterial||y.isMeshPhysicalMaterial)),sizeAttenuation:y.sizeAttenuation===!0,logarithmicDepthBuffer:h,reversedDepthBuffer:ne,skinning:F.isSkinnedMesh===!0,morphTargets:W.morphAttributes.position!==void 0,morphNormals:W.morphAttributes.normal!==void 0,morphColors:W.morphAttributes.color!==void 0,morphTargetsCount:se,morphTextureStride:ae,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:s.numPlanes,numClipIntersection:s.numIntersection,dithering:y.dithering,shadowMapEnabled:t.shadowMap.enabled&&P.length>0,shadowMapType:t.shadowMap.type,toneMapping:We,decodeVideoTexture:Le&&y.map.isVideoTexture===!0&&rt.getTransfer(y.map.colorSpace)===lt,decodeVideoTextureEmissive:Re&&y.emissiveMap.isVideoTexture===!0&&rt.getTransfer(y.emissiveMap.colorSpace)===lt,premultipliedAlpha:y.premultipliedAlpha,doubleSided:y.side===ri,flipSided:y.side===en,useDepthPacking:y.depthPacking>=0,depthPacking:y.depthPacking||0,index0AttributeName:y.index0AttributeName,extensionClipCullDistance:Ie&&y.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ie&&y.extensions.multiDraw===!0||ye)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:y.customProgramCacheKey()};return mt.vertexUv1s=c.has(1),mt.vertexUv2s=c.has(2),mt.vertexUv3s=c.has(3),c.clear(),mt}function v(y){const C=[];if(y.shaderID?C.push(y.shaderID):(C.push(y.customVertexShaderID),C.push(y.customFragmentShaderID)),y.defines!==void 0)for(const P in y.defines)C.push(P),C.push(y.defines[P]);return y.isRawShaderMaterial===!1&&(d(C,y),m(C,y),C.push(t.outputColorSpace)),C.push(y.customProgramCacheKey),C.join()}function d(y,C){y.push(C.precision),y.push(C.outputColorSpace),y.push(C.envMapMode),y.push(C.envMapCubeUVHeight),y.push(C.mapUv),y.push(C.alphaMapUv),y.push(C.lightMapUv),y.push(C.aoMapUv),y.push(C.bumpMapUv),y.push(C.normalMapUv),y.push(C.displacementMapUv),y.push(C.emissiveMapUv),y.push(C.metalnessMapUv),y.push(C.roughnessMapUv),y.push(C.anisotropyMapUv),y.push(C.clearcoatMapUv),y.push(C.clearcoatNormalMapUv),y.push(C.clearcoatRoughnessMapUv),y.push(C.iridescenceMapUv),y.push(C.iridescenceThicknessMapUv),y.push(C.sheenColorMapUv),y.push(C.sheenRoughnessMapUv),y.push(C.specularMapUv),y.push(C.specularColorMapUv),y.push(C.specularIntensityMapUv),y.push(C.transmissionMapUv),y.push(C.thicknessMapUv),y.push(C.combine),y.push(C.fogExp2),y.push(C.sizeAttenuation),y.push(C.morphTargetsCount),y.push(C.morphAttributeCount),y.push(C.numDirLights),y.push(C.numPointLights),y.push(C.numSpotLights),y.push(C.numSpotLightMaps),y.push(C.numHemiLights),y.push(C.numRectAreaLights),y.push(C.numDirLightShadows),y.push(C.numPointLightShadows),y.push(C.numSpotLightShadows),y.push(C.numSpotLightShadowsWithMaps),y.push(C.numLightProbes),y.push(C.shadowMapType),y.push(C.toneMapping),y.push(C.numClippingPlanes),y.push(C.numClipIntersection),y.push(C.depthPacking)}function m(y,C){o.disableAll(),C.instancing&&o.enable(0),C.instancingColor&&o.enable(1),C.instancingMorph&&o.enable(2),C.matcap&&o.enable(3),C.envMap&&o.enable(4),C.normalMapObjectSpace&&o.enable(5),C.normalMapTangentSpace&&o.enable(6),C.clearcoat&&o.enable(7),C.iridescence&&o.enable(8),C.alphaTest&&o.enable(9),C.vertexColors&&o.enable(10),C.vertexAlphas&&o.enable(11),C.vertexUv1s&&o.enable(12),C.vertexUv2s&&o.enable(13),C.vertexUv3s&&o.enable(14),C.vertexTangents&&o.enable(15),C.anisotropy&&o.enable(16),C.alphaHash&&o.enable(17),C.batching&&o.enable(18),C.dispersion&&o.enable(19),C.batchingColor&&o.enable(20),C.gradientMap&&o.enable(21),y.push(o.mask),o.disableAll(),C.fog&&o.enable(0),C.useFog&&o.enable(1),C.flatShading&&o.enable(2),C.logarithmicDepthBuffer&&o.enable(3),C.reversedDepthBuffer&&o.enable(4),C.skinning&&o.enable(5),C.morphTargets&&o.enable(6),C.morphNormals&&o.enable(7),C.morphColors&&o.enable(8),C.premultipliedAlpha&&o.enable(9),C.shadowMapEnabled&&o.enable(10),C.doubleSided&&o.enable(11),C.flipSided&&o.enable(12),C.useDepthPacking&&o.enable(13),C.dithering&&o.enable(14),C.transmission&&o.enable(15),C.sheen&&o.enable(16),C.opaque&&o.enable(17),C.pointsUvs&&o.enable(18),C.decodeVideoTexture&&o.enable(19),C.decodeVideoTextureEmissive&&o.enable(20),C.alphaToCoverage&&o.enable(21),y.push(o.mask)}function _(y){const C=g[y.type];let P;if(C){const I=ni[C];P=RS.clone(I.uniforms)}else P=y.uniforms;return P}function b(y,C){let P=p.get(C);return P!==void 0?++P.usedTimes:(P=new sT(t,C,y,r),u.push(P),p.set(C,P)),P}function w(y){if(--y.usedTimes===0){const C=u.indexOf(y);u[C]=u[u.length-1],u.pop(),p.delete(y.cacheKey),y.destroy()}}function A(y){a.remove(y)}function E(){a.dispose()}return{getParameters:M,getProgramCacheKey:v,getUniforms:_,acquireProgram:b,releaseProgram:w,releaseShaderCache:A,programs:u,dispose:E}}function uT(){let t=new WeakMap;function e(o){return t.has(o)}function n(o){let a=t.get(o);return a===void 0&&(a={},t.set(o,a)),a}function i(o){t.delete(o)}function r(o,a,c){t.get(o)[a]=c}function s(){t=new WeakMap}return{has:e,get:n,remove:i,update:r,dispose:s}}function dT(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.material.id!==e.material.id?t.material.id-e.material.id:t.materialVariant!==e.materialVariant?t.materialVariant-e.materialVariant:t.z!==e.z?t.z-e.z:t.id-e.id}function m0(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function g0(){const t=[];let e=0;const n=[],i=[],r=[];function s(){e=0,n.length=0,i.length=0,r.length=0}function o(f){let g=0;return f.isInstancedMesh&&(g+=2),f.isSkinnedMesh&&(g+=1),g}function a(f,g,x,M,v,d){let m=t[e];return m===void 0?(m={id:f.id,object:f,geometry:g,material:x,materialVariant:o(f),groupOrder:M,renderOrder:f.renderOrder,z:v,group:d},t[e]=m):(m.id=f.id,m.object=f,m.geometry=g,m.material=x,m.materialVariant=o(f),m.groupOrder=M,m.renderOrder=f.renderOrder,m.z=v,m.group=d),e++,m}function c(f,g,x,M,v,d){const m=a(f,g,x,M,v,d);x.transmission>0?i.push(m):x.transparent===!0?r.push(m):n.push(m)}function u(f,g,x,M,v,d){const m=a(f,g,x,M,v,d);x.transmission>0?i.unshift(m):x.transparent===!0?r.unshift(m):n.unshift(m)}function p(f,g){n.length>1&&n.sort(f||dT),i.length>1&&i.sort(g||m0),r.length>1&&r.sort(g||m0)}function h(){for(let f=e,g=t.length;f<g;f++){const x=t[f];if(x.id===null)break;x.id=null,x.object=null,x.geometry=null,x.material=null,x.group=null}}return{opaque:n,transmissive:i,transparent:r,init:s,push:c,unshift:u,finish:h,sort:p}}function fT(){let t=new WeakMap;function e(i,r){const s=t.get(i);let o;return s===void 0?(o=new g0,t.set(i,[o])):r>=s.length?(o=new g0,s.push(o)):o=s[r],o}function n(){t=new WeakMap}return{get:e,dispose:n}}function pT(){const t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={direction:new H,color:new tt};break;case"SpotLight":n={position:new H,direction:new H,color:new tt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new H,color:new tt,distance:0,decay:0};break;case"HemisphereLight":n={direction:new H,skyColor:new tt,groundColor:new tt};break;case"RectAreaLight":n={color:new tt,position:new H,halfWidth:new H,halfHeight:new H};break}return t[e.id]=n,n}}}function hT(){const t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ze};break;case"SpotLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ze};break;case"PointLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ze,shadowCameraNear:1,shadowCameraFar:1e3};break}return t[e.id]=n,n}}}let mT=0;function gT(t,e){return(e.castShadow?2:0)-(t.castShadow?2:0)+(e.map?1:0)-(t.map?1:0)}function xT(t){const e=new pT,n=hT(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let u=0;u<9;u++)i.probe.push(new H);const r=new H,s=new _t,o=new _t;function a(u){let p=0,h=0,f=0;for(let C=0;C<9;C++)i.probe[C].set(0,0,0);let g=0,x=0,M=0,v=0,d=0,m=0,_=0,b=0,w=0,A=0,E=0;u.sort(gT);for(let C=0,P=u.length;C<P;C++){const I=u[C],F=I.color,B=I.intensity,W=I.distance;let V=null;if(I.shadow&&I.shadow.map&&(I.shadow.map.texture.format===qs?V=I.shadow.map.texture:V=I.shadow.map.depthTexture||I.shadow.map.texture),I.isAmbientLight)p+=F.r*B,h+=F.g*B,f+=F.b*B;else if(I.isLightProbe){for(let G=0;G<9;G++)i.probe[G].addScaledVector(I.sh.coefficients[G],B);E++}else if(I.isDirectionalLight){const G=e.get(I);if(G.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){const z=I.shadow,j=n.get(I);j.shadowIntensity=z.intensity,j.shadowBias=z.bias,j.shadowNormalBias=z.normalBias,j.shadowRadius=z.radius,j.shadowMapSize=z.mapSize,i.directionalShadow[g]=j,i.directionalShadowMap[g]=V,i.directionalShadowMatrix[g]=I.shadow.matrix,m++}i.directional[g]=G,g++}else if(I.isSpotLight){const G=e.get(I);G.position.setFromMatrixPosition(I.matrixWorld),G.color.copy(F).multiplyScalar(B),G.distance=W,G.coneCos=Math.cos(I.angle),G.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),G.decay=I.decay,i.spot[M]=G;const z=I.shadow;if(I.map&&(i.spotLightMap[w]=I.map,w++,z.updateMatrices(I),I.castShadow&&A++),i.spotLightMatrix[M]=z.matrix,I.castShadow){const j=n.get(I);j.shadowIntensity=z.intensity,j.shadowBias=z.bias,j.shadowNormalBias=z.normalBias,j.shadowRadius=z.radius,j.shadowMapSize=z.mapSize,i.spotShadow[M]=j,i.spotShadowMap[M]=V,b++}M++}else if(I.isRectAreaLight){const G=e.get(I);G.color.copy(F).multiplyScalar(B),G.halfWidth.set(I.width*.5,0,0),G.halfHeight.set(0,I.height*.5,0),i.rectArea[v]=G,v++}else if(I.isPointLight){const G=e.get(I);if(G.color.copy(I.color).multiplyScalar(I.intensity),G.distance=I.distance,G.decay=I.decay,I.castShadow){const z=I.shadow,j=n.get(I);j.shadowIntensity=z.intensity,j.shadowBias=z.bias,j.shadowNormalBias=z.normalBias,j.shadowRadius=z.radius,j.shadowMapSize=z.mapSize,j.shadowCameraNear=z.camera.near,j.shadowCameraFar=z.camera.far,i.pointShadow[x]=j,i.pointShadowMap[x]=V,i.pointShadowMatrix[x]=I.shadow.matrix,_++}i.point[x]=G,x++}else if(I.isHemisphereLight){const G=e.get(I);G.skyColor.copy(I.color).multiplyScalar(B),G.groundColor.copy(I.groundColor).multiplyScalar(B),i.hemi[d]=G,d++}}v>0&&(t.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=_e.LTC_FLOAT_1,i.rectAreaLTC2=_e.LTC_FLOAT_2):(i.rectAreaLTC1=_e.LTC_HALF_1,i.rectAreaLTC2=_e.LTC_HALF_2)),i.ambient[0]=p,i.ambient[1]=h,i.ambient[2]=f;const y=i.hash;(y.directionalLength!==g||y.pointLength!==x||y.spotLength!==M||y.rectAreaLength!==v||y.hemiLength!==d||y.numDirectionalShadows!==m||y.numPointShadows!==_||y.numSpotShadows!==b||y.numSpotMaps!==w||y.numLightProbes!==E)&&(i.directional.length=g,i.spot.length=M,i.rectArea.length=v,i.point.length=x,i.hemi.length=d,i.directionalShadow.length=m,i.directionalShadowMap.length=m,i.pointShadow.length=_,i.pointShadowMap.length=_,i.spotShadow.length=b,i.spotShadowMap.length=b,i.directionalShadowMatrix.length=m,i.pointShadowMatrix.length=_,i.spotLightMatrix.length=b+w-A,i.spotLightMap.length=w,i.numSpotLightShadowsWithMaps=A,i.numLightProbes=E,y.directionalLength=g,y.pointLength=x,y.spotLength=M,y.rectAreaLength=v,y.hemiLength=d,y.numDirectionalShadows=m,y.numPointShadows=_,y.numSpotShadows=b,y.numSpotMaps=w,y.numLightProbes=E,i.version=mT++)}function c(u,p){let h=0,f=0,g=0,x=0,M=0;const v=p.matrixWorldInverse;for(let d=0,m=u.length;d<m;d++){const _=u[d];if(_.isDirectionalLight){const b=i.directional[h];b.direction.setFromMatrixPosition(_.matrixWorld),r.setFromMatrixPosition(_.target.matrixWorld),b.direction.sub(r),b.direction.transformDirection(v),h++}else if(_.isSpotLight){const b=i.spot[g];b.position.setFromMatrixPosition(_.matrixWorld),b.position.applyMatrix4(v),b.direction.setFromMatrixPosition(_.matrixWorld),r.setFromMatrixPosition(_.target.matrixWorld),b.direction.sub(r),b.direction.transformDirection(v),g++}else if(_.isRectAreaLight){const b=i.rectArea[x];b.position.setFromMatrixPosition(_.matrixWorld),b.position.applyMatrix4(v),o.identity(),s.copy(_.matrixWorld),s.premultiply(v),o.extractRotation(s),b.halfWidth.set(_.width*.5,0,0),b.halfHeight.set(0,_.height*.5,0),b.halfWidth.applyMatrix4(o),b.halfHeight.applyMatrix4(o),x++}else if(_.isPointLight){const b=i.point[f];b.position.setFromMatrixPosition(_.matrixWorld),b.position.applyMatrix4(v),f++}else if(_.isHemisphereLight){const b=i.hemi[M];b.direction.setFromMatrixPosition(_.matrixWorld),b.direction.transformDirection(v),M++}}}return{setup:a,setupView:c,state:i}}function x0(t){const e=new xT(t),n=[],i=[];function r(p){u.camera=p,n.length=0,i.length=0}function s(p){n.push(p)}function o(p){i.push(p)}function a(){e.setup(n)}function c(p){e.setupView(n,p)}const u={lightsArray:n,shadowsArray:i,camera:null,lights:e,transmissionRenderTarget:{}};return{init:r,state:u,setupLights:a,setupLightsView:c,pushLight:s,pushShadow:o}}function vT(t){let e=new WeakMap;function n(r,s=0){const o=e.get(r);let a;return o===void 0?(a=new x0(t),e.set(r,[a])):s>=o.length?(a=new x0(t),o.push(a)):a=o[s],a}function i(){e=new WeakMap}return{get:n,dispose:i}}const _T=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,yT=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,ST=[new H(1,0,0),new H(-1,0,0),new H(0,1,0),new H(0,-1,0),new H(0,0,1),new H(0,0,-1)],MT=[new H(0,-1,0),new H(0,-1,0),new H(0,0,1),new H(0,0,-1),new H(0,-1,0),new H(0,-1,0)],v0=new _t,So=new H,Xu=new H;function bT(t,e,n){let i=new Xp;const r=new Ze,s=new Ze,o=new wt,a=new FS,c=new NS,u={},p=n.maxTextureSize,h={[pr]:en,[en]:pr,[ri]:ri},f=new hi({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ze},radius:{value:4}},vertexShader:_T,fragmentShader:yT}),g=f.clone();g.defines.HORIZONTAL_PASS=1;const x=new rn;x.setAttribute("position",new Kn(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const M=new pn(x,f),v=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=wl;let d=this.type;this.render=function(A,E,y){if(v.enabled===!1||v.autoUpdate===!1&&v.needsUpdate===!1||A.length===0)return;this.type===vy&&(He("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=wl);const C=t.getRenderTarget(),P=t.getActiveCubeFace(),I=t.getActiveMipmapLevel(),F=t.state;F.setBlending(Ri),F.buffers.depth.getReversed()===!0?F.buffers.color.setClear(0,0,0,0):F.buffers.color.setClear(1,1,1,1),F.buffers.depth.setTest(!0),F.setScissorTest(!1);const B=d!==this.type;B&&E.traverse(function(W){W.material&&(Array.isArray(W.material)?W.material.forEach(V=>V.needsUpdate=!0):W.material.needsUpdate=!0)});for(let W=0,V=A.length;W<V;W++){const G=A[W],z=G.shadow;if(z===void 0){He("WebGLShadowMap:",G,"has no shadow.");continue}if(z.autoUpdate===!1&&z.needsUpdate===!1)continue;r.copy(z.mapSize);const j=z.getFrameExtents();r.multiply(j),s.copy(z.mapSize),(r.x>p||r.y>p)&&(r.x>p&&(s.x=Math.floor(p/j.x),r.x=s.x*j.x,z.mapSize.x=s.x),r.y>p&&(s.y=Math.floor(p/j.y),r.y=s.y*j.y,z.mapSize.y=s.y));const $=t.state.buffers.depth.getReversed();if(z.camera._reversedDepth=$,z.map===null||B===!0){if(z.map!==null&&(z.map.depthTexture!==null&&(z.map.depthTexture.dispose(),z.map.depthTexture=null),z.map.dispose()),this.type===wo){if(G.isPointLight){He("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}z.map=new di(r.x,r.y,{format:qs,type:Ni,minFilter:Qt,magFilter:Qt,generateMipmaps:!1}),z.map.texture.name=G.name+".shadowMap",z.map.depthTexture=new ia(r.x,r.y,oi),z.map.depthTexture.name=G.name+".shadowMapDepth",z.map.depthTexture.format=Ui,z.map.depthTexture.compareFunction=null,z.map.depthTexture.minFilter=Vt,z.map.depthTexture.magFilter=Vt}else G.isPointLight?(z.map=new xx(r.x),z.map.depthTexture=new CS(r.x,fi)):(z.map=new di(r.x,r.y),z.map.depthTexture=new ia(r.x,r.y,fi)),z.map.depthTexture.name=G.name+".shadowMap",z.map.depthTexture.format=Ui,this.type===wl?(z.map.depthTexture.compareFunction=$?Hp:Vp,z.map.depthTexture.minFilter=Qt,z.map.depthTexture.magFilter=Qt):(z.map.depthTexture.compareFunction=null,z.map.depthTexture.minFilter=Vt,z.map.depthTexture.magFilter=Vt);z.camera.updateProjectionMatrix()}const Q=z.map.isWebGLCubeRenderTarget?6:1;for(let se=0;se<Q;se++){if(z.map.isWebGLCubeRenderTarget)t.setRenderTarget(z.map,se),t.clear();else{se===0&&(t.setRenderTarget(z.map),t.clear());const ae=z.getViewport(se);o.set(s.x*ae.x,s.y*ae.y,s.x*ae.z,s.y*ae.w),F.viewport(o)}if(G.isPointLight){const ae=z.camera,Ae=z.matrix,De=G.distance||ae.far;De!==ae.far&&(ae.far=De,ae.updateProjectionMatrix()),So.setFromMatrixPosition(G.matrixWorld),ae.position.copy(So),Xu.copy(ae.position),Xu.add(ST[se]),ae.up.copy(MT[se]),ae.lookAt(Xu),ae.updateMatrixWorld(),Ae.makeTranslation(-So.x,-So.y,-So.z),v0.multiplyMatrices(ae.projectionMatrix,ae.matrixWorldInverse),z._frustum.setFromProjectionMatrix(v0,ae.coordinateSystem,ae.reversedDepth)}else z.updateMatrices(G);i=z.getFrustum(),b(E,y,z.camera,G,this.type)}z.isPointLightShadow!==!0&&this.type===wo&&m(z,y),z.needsUpdate=!1}d=this.type,v.needsUpdate=!1,t.setRenderTarget(C,P,I)};function m(A,E){const y=e.update(M);f.defines.VSM_SAMPLES!==A.blurSamples&&(f.defines.VSM_SAMPLES=A.blurSamples,g.defines.VSM_SAMPLES=A.blurSamples,f.needsUpdate=!0,g.needsUpdate=!0),A.mapPass===null&&(A.mapPass=new di(r.x,r.y,{format:qs,type:Ni})),f.uniforms.shadow_pass.value=A.map.depthTexture,f.uniforms.resolution.value=A.mapSize,f.uniforms.radius.value=A.radius,t.setRenderTarget(A.mapPass),t.clear(),t.renderBufferDirect(E,null,y,f,M,null),g.uniforms.shadow_pass.value=A.mapPass.texture,g.uniforms.resolution.value=A.mapSize,g.uniforms.radius.value=A.radius,t.setRenderTarget(A.map),t.clear(),t.renderBufferDirect(E,null,y,g,M,null)}function _(A,E,y,C){let P=null;const I=y.isPointLight===!0?A.customDistanceMaterial:A.customDepthMaterial;if(I!==void 0)P=I;else if(P=y.isPointLight===!0?c:a,t.localClippingEnabled&&E.clipShadows===!0&&Array.isArray(E.clippingPlanes)&&E.clippingPlanes.length!==0||E.displacementMap&&E.displacementScale!==0||E.alphaMap&&E.alphaTest>0||E.map&&E.alphaTest>0||E.alphaToCoverage===!0){const F=P.uuid,B=E.uuid;let W=u[F];W===void 0&&(W={},u[F]=W);let V=W[B];V===void 0&&(V=P.clone(),W[B]=V,E.addEventListener("dispose",w)),P=V}if(P.visible=E.visible,P.wireframe=E.wireframe,C===wo?P.side=E.shadowSide!==null?E.shadowSide:E.side:P.side=E.shadowSide!==null?E.shadowSide:h[E.side],P.alphaMap=E.alphaMap,P.alphaTest=E.alphaToCoverage===!0?.5:E.alphaTest,P.map=E.map,P.clipShadows=E.clipShadows,P.clippingPlanes=E.clippingPlanes,P.clipIntersection=E.clipIntersection,P.displacementMap=E.displacementMap,P.displacementScale=E.displacementScale,P.displacementBias=E.displacementBias,P.wireframeLinewidth=E.wireframeLinewidth,P.linewidth=E.linewidth,y.isPointLight===!0&&P.isMeshDistanceMaterial===!0){const F=t.properties.get(P);F.light=y}return P}function b(A,E,y,C,P){if(A.visible===!1)return;if(A.layers.test(E.layers)&&(A.isMesh||A.isLine||A.isPoints)&&(A.castShadow||A.receiveShadow&&P===wo)&&(!A.frustumCulled||i.intersectsObject(A))){A.modelViewMatrix.multiplyMatrices(y.matrixWorldInverse,A.matrixWorld);const B=e.update(A),W=A.material;if(Array.isArray(W)){const V=B.groups;for(let G=0,z=V.length;G<z;G++){const j=V[G],$=W[j.materialIndex];if($&&$.visible){const Q=_(A,$,C,P);A.onBeforeShadow(t,A,E,y,B,Q,j),t.renderBufferDirect(y,null,B,Q,A,j),A.onAfterShadow(t,A,E,y,B,Q,j)}}}else if(W.visible){const V=_(A,W,C,P);A.onBeforeShadow(t,A,E,y,B,V,null),t.renderBufferDirect(y,null,B,V,A,null),A.onAfterShadow(t,A,E,y,B,V,null)}}const F=A.children;for(let B=0,W=F.length;B<W;B++)b(F[B],E,y,C,P)}function w(A){A.target.removeEventListener("dispose",w);for(const y in u){const C=u[y],P=A.target.uuid;P in C&&(C[P].dispose(),delete C[P])}}}function ET(t,e){function n(){let U=!1;const xe=new wt;let pe=null;const Ee=new wt(0,0,0,0);return{setMask:function(he){pe!==he&&!U&&(t.colorMask(he,he,he,he),pe=he)},setLocked:function(he){U=he},setClear:function(he,ee,Ie,We,mt){mt===!0&&(he*=We,ee*=We,Ie*=We),xe.set(he,ee,Ie,We),Ee.equals(xe)===!1&&(t.clearColor(he,ee,Ie,We),Ee.copy(xe))},reset:function(){U=!1,pe=null,Ee.set(-1,0,0,0)}}}function i(){let U=!1,xe=!1,pe=null,Ee=null,he=null;return{setReversed:function(ee){if(xe!==ee){const Ie=e.get("EXT_clip_control");ee?Ie.clipControlEXT(Ie.LOWER_LEFT_EXT,Ie.ZERO_TO_ONE_EXT):Ie.clipControlEXT(Ie.LOWER_LEFT_EXT,Ie.NEGATIVE_ONE_TO_ONE_EXT),xe=ee;const We=he;he=null,this.setClear(We)}},getReversed:function(){return xe},setTest:function(ee){ee?q(t.DEPTH_TEST):ne(t.DEPTH_TEST)},setMask:function(ee){pe!==ee&&!U&&(t.depthMask(ee),pe=ee)},setFunc:function(ee){if(xe&&(ee=Zy[ee]),Ee!==ee){switch(ee){case Hd:t.depthFunc(t.NEVER);break;case Gd:t.depthFunc(t.ALWAYS);break;case Wd:t.depthFunc(t.LESS);break;case Ws:t.depthFunc(t.LEQUAL);break;case Xd:t.depthFunc(t.EQUAL);break;case qd:t.depthFunc(t.GEQUAL);break;case $d:t.depthFunc(t.GREATER);break;case Kd:t.depthFunc(t.NOTEQUAL);break;default:t.depthFunc(t.LEQUAL)}Ee=ee}},setLocked:function(ee){U=ee},setClear:function(ee){he!==ee&&(he=ee,xe&&(ee=1-ee),t.clearDepth(ee))},reset:function(){U=!1,pe=null,Ee=null,he=null,xe=!1}}}function r(){let U=!1,xe=null,pe=null,Ee=null,he=null,ee=null,Ie=null,We=null,mt=null;return{setTest:function(at){U||(at?q(t.STENCIL_TEST):ne(t.STENCIL_TEST))},setMask:function(at){xe!==at&&!U&&(t.stencilMask(at),xe=at)},setFunc:function(at,gi,xi){(pe!==at||Ee!==gi||he!==xi)&&(t.stencilFunc(at,gi,xi),pe=at,Ee=gi,he=xi)},setOp:function(at,gi,xi){(ee!==at||Ie!==gi||We!==xi)&&(t.stencilOp(at,gi,xi),ee=at,Ie=gi,We=xi)},setLocked:function(at){U=at},setClear:function(at){mt!==at&&(t.clearStencil(at),mt=at)},reset:function(){U=!1,xe=null,pe=null,Ee=null,he=null,ee=null,Ie=null,We=null,mt=null}}}const s=new n,o=new i,a=new r,c=new WeakMap,u=new WeakMap;let p={},h={},f=new WeakMap,g=[],x=null,M=!1,v=null,d=null,m=null,_=null,b=null,w=null,A=null,E=new tt(0,0,0),y=0,C=!1,P=null,I=null,F=null,B=null,W=null;const V=t.getParameter(t.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let G=!1,z=0;const j=t.getParameter(t.VERSION);j.indexOf("WebGL")!==-1?(z=parseFloat(/^WebGL (\d)/.exec(j)[1]),G=z>=1):j.indexOf("OpenGL ES")!==-1&&(z=parseFloat(/^OpenGL ES (\d)/.exec(j)[1]),G=z>=2);let $=null,Q={};const se=t.getParameter(t.SCISSOR_BOX),ae=t.getParameter(t.VIEWPORT),Ae=new wt().fromArray(se),De=new wt().fromArray(ae);function Oe(U,xe,pe,Ee){const he=new Uint8Array(4),ee=t.createTexture();t.bindTexture(U,ee),t.texParameteri(U,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(U,t.TEXTURE_MAG_FILTER,t.NEAREST);for(let Ie=0;Ie<pe;Ie++)U===t.TEXTURE_3D||U===t.TEXTURE_2D_ARRAY?t.texImage3D(xe,0,t.RGBA,1,1,Ee,0,t.RGBA,t.UNSIGNED_BYTE,he):t.texImage2D(xe+Ie,0,t.RGBA,1,1,0,t.RGBA,t.UNSIGNED_BYTE,he);return ee}const D={};D[t.TEXTURE_2D]=Oe(t.TEXTURE_2D,t.TEXTURE_2D,1),D[t.TEXTURE_CUBE_MAP]=Oe(t.TEXTURE_CUBE_MAP,t.TEXTURE_CUBE_MAP_POSITIVE_X,6),D[t.TEXTURE_2D_ARRAY]=Oe(t.TEXTURE_2D_ARRAY,t.TEXTURE_2D_ARRAY,1,1),D[t.TEXTURE_3D]=Oe(t.TEXTURE_3D,t.TEXTURE_3D,1,1),s.setClear(0,0,0,1),o.setClear(1),a.setClear(0),q(t.DEPTH_TEST),o.setFunc(Ws),le(!1),ze(_m),q(t.CULL_FACE),Ve(Ri);function q(U){p[U]!==!0&&(t.enable(U),p[U]=!0)}function ne(U){p[U]!==!1&&(t.disable(U),p[U]=!1)}function oe(U,xe){return h[U]!==xe?(t.bindFramebuffer(U,xe),h[U]=xe,U===t.DRAW_FRAMEBUFFER&&(h[t.FRAMEBUFFER]=xe),U===t.FRAMEBUFFER&&(h[t.DRAW_FRAMEBUFFER]=xe),!0):!1}function ye(U,xe){let pe=g,Ee=!1;if(U){pe=f.get(xe),pe===void 0&&(pe=[],f.set(xe,pe));const he=U.textures;if(pe.length!==he.length||pe[0]!==t.COLOR_ATTACHMENT0){for(let ee=0,Ie=he.length;ee<Ie;ee++)pe[ee]=t.COLOR_ATTACHMENT0+ee;pe.length=he.length,Ee=!0}}else pe[0]!==t.BACK&&(pe[0]=t.BACK,Ee=!0);Ee&&t.drawBuffers(pe)}function Le(U){return x!==U?(t.useProgram(U),x=U,!0):!1}const ht={[Pr]:t.FUNC_ADD,[yy]:t.FUNC_SUBTRACT,[Sy]:t.FUNC_REVERSE_SUBTRACT};ht[My]=t.MIN,ht[by]=t.MAX;const ve={[Ey]:t.ZERO,[Ty]:t.ONE,[wy]:t.SRC_COLOR,[jd]:t.SRC_ALPHA,[Dy]:t.SRC_ALPHA_SATURATE,[Iy]:t.DST_COLOR,[Ay]:t.DST_ALPHA,[Cy]:t.ONE_MINUS_SRC_COLOR,[Vd]:t.ONE_MINUS_SRC_ALPHA,[Py]:t.ONE_MINUS_DST_COLOR,[Ry]:t.ONE_MINUS_DST_ALPHA,[Ly]:t.CONSTANT_COLOR,[Fy]:t.ONE_MINUS_CONSTANT_COLOR,[Ny]:t.CONSTANT_ALPHA,[Uy]:t.ONE_MINUS_CONSTANT_ALPHA};function Ve(U,xe,pe,Ee,he,ee,Ie,We,mt,at){if(U===Ri){M===!0&&(ne(t.BLEND),M=!1);return}if(M===!1&&(q(t.BLEND),M=!0),U!==_y){if(U!==v||at!==C){if((d!==Pr||b!==Pr)&&(t.blendEquation(t.FUNC_ADD),d=Pr,b=Pr),at)switch(U){case Us:t.blendFuncSeparate(t.ONE,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case Bd:t.blendFunc(t.ONE,t.ONE);break;case ym:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case Sm:t.blendFuncSeparate(t.DST_COLOR,t.ONE_MINUS_SRC_ALPHA,t.ZERO,t.ONE);break;default:it("WebGLState: Invalid blending: ",U);break}else switch(U){case Us:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case Bd:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE,t.ONE,t.ONE);break;case ym:it("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Sm:it("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:it("WebGLState: Invalid blending: ",U);break}m=null,_=null,w=null,A=null,E.set(0,0,0),y=0,v=U,C=at}return}he=he||xe,ee=ee||pe,Ie=Ie||Ee,(xe!==d||he!==b)&&(t.blendEquationSeparate(ht[xe],ht[he]),d=xe,b=he),(pe!==m||Ee!==_||ee!==w||Ie!==A)&&(t.blendFuncSeparate(ve[pe],ve[Ee],ve[ee],ve[Ie]),m=pe,_=Ee,w=ee,A=Ie),(We.equals(E)===!1||mt!==y)&&(t.blendColor(We.r,We.g,We.b,mt),E.copy(We),y=mt),v=U,C=!1}function ie(U,xe){U.side===ri?ne(t.CULL_FACE):q(t.CULL_FACE);let pe=U.side===en;xe&&(pe=!pe),le(pe),U.blending===Us&&U.transparent===!1?Ve(Ri):Ve(U.blending,U.blendEquation,U.blendSrc,U.blendDst,U.blendEquationAlpha,U.blendSrcAlpha,U.blendDstAlpha,U.blendColor,U.blendAlpha,U.premultipliedAlpha),o.setFunc(U.depthFunc),o.setTest(U.depthTest),o.setMask(U.depthWrite),s.setMask(U.colorWrite);const Ee=U.stencilWrite;a.setTest(Ee),Ee&&(a.setMask(U.stencilWriteMask),a.setFunc(U.stencilFunc,U.stencilRef,U.stencilFuncMask),a.setOp(U.stencilFail,U.stencilZFail,U.stencilZPass)),Re(U.polygonOffset,U.polygonOffsetFactor,U.polygonOffsetUnits),U.alphaToCoverage===!0?q(t.SAMPLE_ALPHA_TO_COVERAGE):ne(t.SAMPLE_ALPHA_TO_COVERAGE)}function le(U){P!==U&&(U?t.frontFace(t.CW):t.frontFace(t.CCW),P=U)}function ze(U){U!==gy?(q(t.CULL_FACE),U!==I&&(U===_m?t.cullFace(t.BACK):U===xy?t.cullFace(t.FRONT):t.cullFace(t.FRONT_AND_BACK))):ne(t.CULL_FACE),I=U}function L(U){U!==F&&(G&&t.lineWidth(U),F=U)}function Re(U,xe,pe){U?(q(t.POLYGON_OFFSET_FILL),(B!==xe||W!==pe)&&(B=xe,W=pe,o.getReversed()&&(xe=-xe),t.polygonOffset(xe,pe))):ne(t.POLYGON_OFFSET_FILL)}function Ge(U){U?q(t.SCISSOR_TEST):ne(t.SCISSOR_TEST)}function $e(U){U===void 0&&(U=t.TEXTURE0+V-1),$!==U&&(t.activeTexture(U),$=U)}function Te(U,xe,pe){pe===void 0&&($===null?pe=t.TEXTURE0+V-1:pe=$);let Ee=Q[pe];Ee===void 0&&(Ee={type:void 0,texture:void 0},Q[pe]=Ee),(Ee.type!==U||Ee.texture!==xe)&&($!==pe&&(t.activeTexture(pe),$=pe),t.bindTexture(U,xe||D[U]),Ee.type=U,Ee.texture=xe)}function R(){const U=Q[$];U!==void 0&&U.type!==void 0&&(t.bindTexture(U.type,null),U.type=void 0,U.texture=void 0)}function S(){try{t.compressedTexImage2D(...arguments)}catch(U){it("WebGLState:",U)}}function k(){try{t.compressedTexImage3D(...arguments)}catch(U){it("WebGLState:",U)}}function te(){try{t.texSubImage2D(...arguments)}catch(U){it("WebGLState:",U)}}function re(){try{t.texSubImage3D(...arguments)}catch(U){it("WebGLState:",U)}}function J(){try{t.compressedTexSubImage2D(...arguments)}catch(U){it("WebGLState:",U)}}function be(){try{t.compressedTexSubImage3D(...arguments)}catch(U){it("WebGLState:",U)}}function ge(){try{t.texStorage2D(...arguments)}catch(U){it("WebGLState:",U)}}function Ue(){try{t.texStorage3D(...arguments)}catch(U){it("WebGLState:",U)}}function Be(){try{t.texImage2D(...arguments)}catch(U){it("WebGLState:",U)}}function de(){try{t.texImage3D(...arguments)}catch(U){it("WebGLState:",U)}}function me(U){Ae.equals(U)===!1&&(t.scissor(U.x,U.y,U.z,U.w),Ae.copy(U))}function N(U){De.equals(U)===!1&&(t.viewport(U.x,U.y,U.z,U.w),De.copy(U))}function ce(U,xe){let pe=u.get(xe);pe===void 0&&(pe=new WeakMap,u.set(xe,pe));let Ee=pe.get(U);Ee===void 0&&(Ee=t.getUniformBlockIndex(xe,U.name),pe.set(U,Ee))}function ue(U,xe){const Ee=u.get(xe).get(U);c.get(xe)!==Ee&&(t.uniformBlockBinding(xe,Ee,U.__bindingPointIndex),c.set(xe,Ee))}function je(){t.disable(t.BLEND),t.disable(t.CULL_FACE),t.disable(t.DEPTH_TEST),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SCISSOR_TEST),t.disable(t.STENCIL_TEST),t.disable(t.SAMPLE_ALPHA_TO_COVERAGE),t.blendEquation(t.FUNC_ADD),t.blendFunc(t.ONE,t.ZERO),t.blendFuncSeparate(t.ONE,t.ZERO,t.ONE,t.ZERO),t.blendColor(0,0,0,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(t.LESS),o.setReversed(!1),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(t.ALWAYS,0,4294967295),t.stencilOp(t.KEEP,t.KEEP,t.KEEP),t.clearStencil(0),t.cullFace(t.BACK),t.frontFace(t.CCW),t.polygonOffset(0,0),t.activeTexture(t.TEXTURE0),t.bindFramebuffer(t.FRAMEBUFFER,null),t.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),t.bindFramebuffer(t.READ_FRAMEBUFFER,null),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),p={},$=null,Q={},h={},f=new WeakMap,g=[],x=null,M=!1,v=null,d=null,m=null,_=null,b=null,w=null,A=null,E=new tt(0,0,0),y=0,C=!1,P=null,I=null,F=null,B=null,W=null,Ae.set(0,0,t.canvas.width,t.canvas.height),De.set(0,0,t.canvas.width,t.canvas.height),s.reset(),o.reset(),a.reset()}return{buffers:{color:s,depth:o,stencil:a},enable:q,disable:ne,bindFramebuffer:oe,drawBuffers:ye,useProgram:Le,setBlending:Ve,setMaterial:ie,setFlipSided:le,setCullFace:ze,setLineWidth:L,setPolygonOffset:Re,setScissorTest:Ge,activeTexture:$e,bindTexture:Te,unbindTexture:R,compressedTexImage2D:S,compressedTexImage3D:k,texImage2D:Be,texImage3D:de,updateUBOMapping:ce,uniformBlockBinding:ue,texStorage2D:ge,texStorage3D:Ue,texSubImage2D:te,texSubImage3D:re,compressedTexSubImage2D:J,compressedTexSubImage3D:be,scissor:me,viewport:N,reset:je}}function TT(t,e,n,i,r,s,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),u=new Ze,p=new WeakMap;let h;const f=new WeakMap;let g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,S){return g?new OffscreenCanvas(R,S):sc("canvas")}function M(R,S,k){let te=1;const re=Te(R);if((re.width>k||re.height>k)&&(te=k/Math.max(re.width,re.height)),te<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){const J=Math.floor(te*re.width),be=Math.floor(te*re.height);h===void 0&&(h=x(J,be));const ge=S?x(J,be):h;return ge.width=J,ge.height=be,ge.getContext("2d").drawImage(R,0,0,J,be),He("WebGLRenderer: Texture has been resized from ("+re.width+"x"+re.height+") to ("+J+"x"+be+")."),ge}else return"data"in R&&He("WebGLRenderer: Image in DataTexture is too big ("+re.width+"x"+re.height+")."),R;return R}function v(R){return R.generateMipmaps}function d(R){t.generateMipmap(R)}function m(R){return R.isWebGLCubeRenderTarget?t.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?t.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?t.TEXTURE_2D_ARRAY:t.TEXTURE_2D}function _(R,S,k,te,re=!1){if(R!==null){if(t[R]!==void 0)return t[R];He("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let J=S;if(S===t.RED&&(k===t.FLOAT&&(J=t.R32F),k===t.HALF_FLOAT&&(J=t.R16F),k===t.UNSIGNED_BYTE&&(J=t.R8)),S===t.RED_INTEGER&&(k===t.UNSIGNED_BYTE&&(J=t.R8UI),k===t.UNSIGNED_SHORT&&(J=t.R16UI),k===t.UNSIGNED_INT&&(J=t.R32UI),k===t.BYTE&&(J=t.R8I),k===t.SHORT&&(J=t.R16I),k===t.INT&&(J=t.R32I)),S===t.RG&&(k===t.FLOAT&&(J=t.RG32F),k===t.HALF_FLOAT&&(J=t.RG16F),k===t.UNSIGNED_BYTE&&(J=t.RG8)),S===t.RG_INTEGER&&(k===t.UNSIGNED_BYTE&&(J=t.RG8UI),k===t.UNSIGNED_SHORT&&(J=t.RG16UI),k===t.UNSIGNED_INT&&(J=t.RG32UI),k===t.BYTE&&(J=t.RG8I),k===t.SHORT&&(J=t.RG16I),k===t.INT&&(J=t.RG32I)),S===t.RGB_INTEGER&&(k===t.UNSIGNED_BYTE&&(J=t.RGB8UI),k===t.UNSIGNED_SHORT&&(J=t.RGB16UI),k===t.UNSIGNED_INT&&(J=t.RGB32UI),k===t.BYTE&&(J=t.RGB8I),k===t.SHORT&&(J=t.RGB16I),k===t.INT&&(J=t.RGB32I)),S===t.RGBA_INTEGER&&(k===t.UNSIGNED_BYTE&&(J=t.RGBA8UI),k===t.UNSIGNED_SHORT&&(J=t.RGBA16UI),k===t.UNSIGNED_INT&&(J=t.RGBA32UI),k===t.BYTE&&(J=t.RGBA8I),k===t.SHORT&&(J=t.RGBA16I),k===t.INT&&(J=t.RGBA32I)),S===t.RGB&&(k===t.UNSIGNED_INT_5_9_9_9_REV&&(J=t.RGB9_E5),k===t.UNSIGNED_INT_10F_11F_11F_REV&&(J=t.R11F_G11F_B10F)),S===t.RGBA){const be=re?rc:rt.getTransfer(te);k===t.FLOAT&&(J=t.RGBA32F),k===t.HALF_FLOAT&&(J=t.RGBA16F),k===t.UNSIGNED_BYTE&&(J=be===lt?t.SRGB8_ALPHA8:t.RGBA8),k===t.UNSIGNED_SHORT_4_4_4_4&&(J=t.RGBA4),k===t.UNSIGNED_SHORT_5_5_5_1&&(J=t.RGB5_A1)}return(J===t.R16F||J===t.R32F||J===t.RG16F||J===t.RG32F||J===t.RGBA16F||J===t.RGBA32F)&&e.get("EXT_color_buffer_float"),J}function b(R,S){let k;return R?S===null||S===fi||S===ta?k=t.DEPTH24_STENCIL8:S===oi?k=t.DEPTH32F_STENCIL8:S===ea&&(k=t.DEPTH24_STENCIL8,He("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):S===null||S===fi||S===ta?k=t.DEPTH_COMPONENT24:S===oi?k=t.DEPTH_COMPONENT32F:S===ea&&(k=t.DEPTH_COMPONENT16),k}function w(R,S){return v(R)===!0||R.isFramebufferTexture&&R.minFilter!==Vt&&R.minFilter!==Qt?Math.log2(Math.max(S.width,S.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?S.mipmaps.length:1}function A(R){const S=R.target;S.removeEventListener("dispose",A),y(S),S.isVideoTexture&&p.delete(S)}function E(R){const S=R.target;S.removeEventListener("dispose",E),P(S)}function y(R){const S=i.get(R);if(S.__webglInit===void 0)return;const k=R.source,te=f.get(k);if(te){const re=te[S.__cacheKey];re.usedTimes--,re.usedTimes===0&&C(R),Object.keys(te).length===0&&f.delete(k)}i.remove(R)}function C(R){const S=i.get(R);t.deleteTexture(S.__webglTexture);const k=R.source,te=f.get(k);delete te[S.__cacheKey],o.memory.textures--}function P(R){const S=i.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),i.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let te=0;te<6;te++){if(Array.isArray(S.__webglFramebuffer[te]))for(let re=0;re<S.__webglFramebuffer[te].length;re++)t.deleteFramebuffer(S.__webglFramebuffer[te][re]);else t.deleteFramebuffer(S.__webglFramebuffer[te]);S.__webglDepthbuffer&&t.deleteRenderbuffer(S.__webglDepthbuffer[te])}else{if(Array.isArray(S.__webglFramebuffer))for(let te=0;te<S.__webglFramebuffer.length;te++)t.deleteFramebuffer(S.__webglFramebuffer[te]);else t.deleteFramebuffer(S.__webglFramebuffer);if(S.__webglDepthbuffer&&t.deleteRenderbuffer(S.__webglDepthbuffer),S.__webglMultisampledFramebuffer&&t.deleteFramebuffer(S.__webglMultisampledFramebuffer),S.__webglColorRenderbuffer)for(let te=0;te<S.__webglColorRenderbuffer.length;te++)S.__webglColorRenderbuffer[te]&&t.deleteRenderbuffer(S.__webglColorRenderbuffer[te]);S.__webglDepthRenderbuffer&&t.deleteRenderbuffer(S.__webglDepthRenderbuffer)}const k=R.textures;for(let te=0,re=k.length;te<re;te++){const J=i.get(k[te]);J.__webglTexture&&(t.deleteTexture(J.__webglTexture),o.memory.textures--),i.remove(k[te])}i.remove(R)}let I=0;function F(){I=0}function B(){const R=I;return R>=r.maxTextures&&He("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+r.maxTextures),I+=1,R}function W(R){const S=[];return S.push(R.wrapS),S.push(R.wrapT),S.push(R.wrapR||0),S.push(R.magFilter),S.push(R.minFilter),S.push(R.anisotropy),S.push(R.internalFormat),S.push(R.format),S.push(R.type),S.push(R.generateMipmaps),S.push(R.premultiplyAlpha),S.push(R.flipY),S.push(R.unpackAlignment),S.push(R.colorSpace),S.join()}function V(R,S){const k=i.get(R);if(R.isVideoTexture&&Ge(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&k.__version!==R.version){const te=R.image;if(te===null)He("WebGLRenderer: Texture marked for update but no image data found.");else if(te.complete===!1)He("WebGLRenderer: Texture marked for update but image is incomplete");else{D(k,R,S);return}}else R.isExternalTexture&&(k.__webglTexture=R.sourceTexture?R.sourceTexture:null);n.bindTexture(t.TEXTURE_2D,k.__webglTexture,t.TEXTURE0+S)}function G(R,S){const k=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&k.__version!==R.version){D(k,R,S);return}else R.isExternalTexture&&(k.__webglTexture=R.sourceTexture?R.sourceTexture:null);n.bindTexture(t.TEXTURE_2D_ARRAY,k.__webglTexture,t.TEXTURE0+S)}function z(R,S){const k=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&k.__version!==R.version){D(k,R,S);return}n.bindTexture(t.TEXTURE_3D,k.__webglTexture,t.TEXTURE0+S)}function j(R,S){const k=i.get(R);if(R.isCubeDepthTexture!==!0&&R.version>0&&k.__version!==R.version){q(k,R,S);return}n.bindTexture(t.TEXTURE_CUBE_MAP,k.__webglTexture,t.TEXTURE0+S)}const $={[Yd]:t.REPEAT,[Ci]:t.CLAMP_TO_EDGE,[Zd]:t.MIRRORED_REPEAT},Q={[Vt]:t.NEAREST,[zy]:t.NEAREST_MIPMAP_NEAREST,[ka]:t.NEAREST_MIPMAP_LINEAR,[Qt]:t.LINEAR,[pu]:t.LINEAR_MIPMAP_NEAREST,[Ur]:t.LINEAR_MIPMAP_LINEAR},se={[Vy]:t.NEVER,[qy]:t.ALWAYS,[Hy]:t.LESS,[Vp]:t.LEQUAL,[Gy]:t.EQUAL,[Hp]:t.GEQUAL,[Wy]:t.GREATER,[Xy]:t.NOTEQUAL};function ae(R,S){if(S.type===oi&&e.has("OES_texture_float_linear")===!1&&(S.magFilter===Qt||S.magFilter===pu||S.magFilter===ka||S.magFilter===Ur||S.minFilter===Qt||S.minFilter===pu||S.minFilter===ka||S.minFilter===Ur)&&He("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),t.texParameteri(R,t.TEXTURE_WRAP_S,$[S.wrapS]),t.texParameteri(R,t.TEXTURE_WRAP_T,$[S.wrapT]),(R===t.TEXTURE_3D||R===t.TEXTURE_2D_ARRAY)&&t.texParameteri(R,t.TEXTURE_WRAP_R,$[S.wrapR]),t.texParameteri(R,t.TEXTURE_MAG_FILTER,Q[S.magFilter]),t.texParameteri(R,t.TEXTURE_MIN_FILTER,Q[S.minFilter]),S.compareFunction&&(t.texParameteri(R,t.TEXTURE_COMPARE_MODE,t.COMPARE_REF_TO_TEXTURE),t.texParameteri(R,t.TEXTURE_COMPARE_FUNC,se[S.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(S.magFilter===Vt||S.minFilter!==ka&&S.minFilter!==Ur||S.type===oi&&e.has("OES_texture_float_linear")===!1)return;if(S.anisotropy>1||i.get(S).__currentAnisotropy){const k=e.get("EXT_texture_filter_anisotropic");t.texParameterf(R,k.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(S.anisotropy,r.getMaxAnisotropy())),i.get(S).__currentAnisotropy=S.anisotropy}}}function Ae(R,S){let k=!1;R.__webglInit===void 0&&(R.__webglInit=!0,S.addEventListener("dispose",A));const te=S.source;let re=f.get(te);re===void 0&&(re={},f.set(te,re));const J=W(S);if(J!==R.__cacheKey){re[J]===void 0&&(re[J]={texture:t.createTexture(),usedTimes:0},o.memory.textures++,k=!0),re[J].usedTimes++;const be=re[R.__cacheKey];be!==void 0&&(re[R.__cacheKey].usedTimes--,be.usedTimes===0&&C(S)),R.__cacheKey=J,R.__webglTexture=re[J].texture}return k}function De(R,S,k){return Math.floor(Math.floor(R/k)/S)}function Oe(R,S,k,te){const J=R.updateRanges;if(J.length===0)n.texSubImage2D(t.TEXTURE_2D,0,0,0,S.width,S.height,k,te,S.data);else{J.sort((de,me)=>de.start-me.start);let be=0;for(let de=1;de<J.length;de++){const me=J[be],N=J[de],ce=me.start+me.count,ue=De(N.start,S.width,4),je=De(me.start,S.width,4);N.start<=ce+1&&ue===je&&De(N.start+N.count-1,S.width,4)===ue?me.count=Math.max(me.count,N.start+N.count-me.start):(++be,J[be]=N)}J.length=be+1;const ge=t.getParameter(t.UNPACK_ROW_LENGTH),Ue=t.getParameter(t.UNPACK_SKIP_PIXELS),Be=t.getParameter(t.UNPACK_SKIP_ROWS);t.pixelStorei(t.UNPACK_ROW_LENGTH,S.width);for(let de=0,me=J.length;de<me;de++){const N=J[de],ce=Math.floor(N.start/4),ue=Math.ceil(N.count/4),je=ce%S.width,U=Math.floor(ce/S.width),xe=ue,pe=1;t.pixelStorei(t.UNPACK_SKIP_PIXELS,je),t.pixelStorei(t.UNPACK_SKIP_ROWS,U),n.texSubImage2D(t.TEXTURE_2D,0,je,U,xe,pe,k,te,S.data)}R.clearUpdateRanges(),t.pixelStorei(t.UNPACK_ROW_LENGTH,ge),t.pixelStorei(t.UNPACK_SKIP_PIXELS,Ue),t.pixelStorei(t.UNPACK_SKIP_ROWS,Be)}}function D(R,S,k){let te=t.TEXTURE_2D;(S.isDataArrayTexture||S.isCompressedArrayTexture)&&(te=t.TEXTURE_2D_ARRAY),S.isData3DTexture&&(te=t.TEXTURE_3D);const re=Ae(R,S),J=S.source;n.bindTexture(te,R.__webglTexture,t.TEXTURE0+k);const be=i.get(J);if(J.version!==be.__version||re===!0){n.activeTexture(t.TEXTURE0+k);const ge=rt.getPrimaries(rt.workingColorSpace),Ue=S.colorSpace===Zi?null:rt.getPrimaries(S.colorSpace),Be=S.colorSpace===Zi||ge===Ue?t.NONE:t.BROWSER_DEFAULT_WEBGL;t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,S.flipY),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),t.pixelStorei(t.UNPACK_ALIGNMENT,S.unpackAlignment),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,Be);let de=M(S.image,!1,r.maxTextureSize);de=$e(S,de);const me=s.convert(S.format,S.colorSpace),N=s.convert(S.type);let ce=_(S.internalFormat,me,N,S.colorSpace,S.isVideoTexture);ae(te,S);let ue;const je=S.mipmaps,U=S.isVideoTexture!==!0,xe=be.__version===void 0||re===!0,pe=J.dataReady,Ee=w(S,de);if(S.isDepthTexture)ce=b(S.format===kr,S.type),xe&&(U?n.texStorage2D(t.TEXTURE_2D,1,ce,de.width,de.height):n.texImage2D(t.TEXTURE_2D,0,ce,de.width,de.height,0,me,N,null));else if(S.isDataTexture)if(je.length>0){U&&xe&&n.texStorage2D(t.TEXTURE_2D,Ee,ce,je[0].width,je[0].height);for(let he=0,ee=je.length;he<ee;he++)ue=je[he],U?pe&&n.texSubImage2D(t.TEXTURE_2D,he,0,0,ue.width,ue.height,me,N,ue.data):n.texImage2D(t.TEXTURE_2D,he,ce,ue.width,ue.height,0,me,N,ue.data);S.generateMipmaps=!1}else U?(xe&&n.texStorage2D(t.TEXTURE_2D,Ee,ce,de.width,de.height),pe&&Oe(S,de,me,N)):n.texImage2D(t.TEXTURE_2D,0,ce,de.width,de.height,0,me,N,de.data);else if(S.isCompressedTexture)if(S.isCompressedArrayTexture){U&&xe&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Ee,ce,je[0].width,je[0].height,de.depth);for(let he=0,ee=je.length;he<ee;he++)if(ue=je[he],S.format!==Xn)if(me!==null)if(U){if(pe)if(S.layerUpdates.size>0){const Ie=Km(ue.width,ue.height,S.format,S.type);for(const We of S.layerUpdates){const mt=ue.data.subarray(We*Ie/ue.data.BYTES_PER_ELEMENT,(We+1)*Ie/ue.data.BYTES_PER_ELEMENT);n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,he,0,0,We,ue.width,ue.height,1,me,mt)}S.clearLayerUpdates()}else n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,he,0,0,0,ue.width,ue.height,de.depth,me,ue.data)}else n.compressedTexImage3D(t.TEXTURE_2D_ARRAY,he,ce,ue.width,ue.height,de.depth,0,ue.data,0,0);else He("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else U?pe&&n.texSubImage3D(t.TEXTURE_2D_ARRAY,he,0,0,0,ue.width,ue.height,de.depth,me,N,ue.data):n.texImage3D(t.TEXTURE_2D_ARRAY,he,ce,ue.width,ue.height,de.depth,0,me,N,ue.data)}else{U&&xe&&n.texStorage2D(t.TEXTURE_2D,Ee,ce,je[0].width,je[0].height);for(let he=0,ee=je.length;he<ee;he++)ue=je[he],S.format!==Xn?me!==null?U?pe&&n.compressedTexSubImage2D(t.TEXTURE_2D,he,0,0,ue.width,ue.height,me,ue.data):n.compressedTexImage2D(t.TEXTURE_2D,he,ce,ue.width,ue.height,0,ue.data):He("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):U?pe&&n.texSubImage2D(t.TEXTURE_2D,he,0,0,ue.width,ue.height,me,N,ue.data):n.texImage2D(t.TEXTURE_2D,he,ce,ue.width,ue.height,0,me,N,ue.data)}else if(S.isDataArrayTexture)if(U){if(xe&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Ee,ce,de.width,de.height,de.depth),pe)if(S.layerUpdates.size>0){const he=Km(de.width,de.height,S.format,S.type);for(const ee of S.layerUpdates){const Ie=de.data.subarray(ee*he/de.data.BYTES_PER_ELEMENT,(ee+1)*he/de.data.BYTES_PER_ELEMENT);n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,ee,de.width,de.height,1,me,N,Ie)}S.clearLayerUpdates()}else n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,0,de.width,de.height,de.depth,me,N,de.data)}else n.texImage3D(t.TEXTURE_2D_ARRAY,0,ce,de.width,de.height,de.depth,0,me,N,de.data);else if(S.isData3DTexture)U?(xe&&n.texStorage3D(t.TEXTURE_3D,Ee,ce,de.width,de.height,de.depth),pe&&n.texSubImage3D(t.TEXTURE_3D,0,0,0,0,de.width,de.height,de.depth,me,N,de.data)):n.texImage3D(t.TEXTURE_3D,0,ce,de.width,de.height,de.depth,0,me,N,de.data);else if(S.isFramebufferTexture){if(xe)if(U)n.texStorage2D(t.TEXTURE_2D,Ee,ce,de.width,de.height);else{let he=de.width,ee=de.height;for(let Ie=0;Ie<Ee;Ie++)n.texImage2D(t.TEXTURE_2D,Ie,ce,he,ee,0,me,N,null),he>>=1,ee>>=1}}else if(je.length>0){if(U&&xe){const he=Te(je[0]);n.texStorage2D(t.TEXTURE_2D,Ee,ce,he.width,he.height)}for(let he=0,ee=je.length;he<ee;he++)ue=je[he],U?pe&&n.texSubImage2D(t.TEXTURE_2D,he,0,0,me,N,ue):n.texImage2D(t.TEXTURE_2D,he,ce,me,N,ue);S.generateMipmaps=!1}else if(U){if(xe){const he=Te(de);n.texStorage2D(t.TEXTURE_2D,Ee,ce,he.width,he.height)}pe&&n.texSubImage2D(t.TEXTURE_2D,0,0,0,me,N,de)}else n.texImage2D(t.TEXTURE_2D,0,ce,me,N,de);v(S)&&d(te),be.__version=J.version,S.onUpdate&&S.onUpdate(S)}R.__version=S.version}function q(R,S,k){if(S.image.length!==6)return;const te=Ae(R,S),re=S.source;n.bindTexture(t.TEXTURE_CUBE_MAP,R.__webglTexture,t.TEXTURE0+k);const J=i.get(re);if(re.version!==J.__version||te===!0){n.activeTexture(t.TEXTURE0+k);const be=rt.getPrimaries(rt.workingColorSpace),ge=S.colorSpace===Zi?null:rt.getPrimaries(S.colorSpace),Ue=S.colorSpace===Zi||be===ge?t.NONE:t.BROWSER_DEFAULT_WEBGL;t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,S.flipY),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),t.pixelStorei(t.UNPACK_ALIGNMENT,S.unpackAlignment),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ue);const Be=S.isCompressedTexture||S.image[0].isCompressedTexture,de=S.image[0]&&S.image[0].isDataTexture,me=[];for(let ee=0;ee<6;ee++)!Be&&!de?me[ee]=M(S.image[ee],!0,r.maxCubemapSize):me[ee]=de?S.image[ee].image:S.image[ee],me[ee]=$e(S,me[ee]);const N=me[0],ce=s.convert(S.format,S.colorSpace),ue=s.convert(S.type),je=_(S.internalFormat,ce,ue,S.colorSpace),U=S.isVideoTexture!==!0,xe=J.__version===void 0||te===!0,pe=re.dataReady;let Ee=w(S,N);ae(t.TEXTURE_CUBE_MAP,S);let he;if(Be){U&&xe&&n.texStorage2D(t.TEXTURE_CUBE_MAP,Ee,je,N.width,N.height);for(let ee=0;ee<6;ee++){he=me[ee].mipmaps;for(let Ie=0;Ie<he.length;Ie++){const We=he[Ie];S.format!==Xn?ce!==null?U?pe&&n.compressedTexSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie,0,0,We.width,We.height,ce,We.data):n.compressedTexImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie,je,We.width,We.height,0,We.data):He("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):U?pe&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie,0,0,We.width,We.height,ce,ue,We.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie,je,We.width,We.height,0,ce,ue,We.data)}}}else{if(he=S.mipmaps,U&&xe){he.length>0&&Ee++;const ee=Te(me[0]);n.texStorage2D(t.TEXTURE_CUBE_MAP,Ee,je,ee.width,ee.height)}for(let ee=0;ee<6;ee++)if(de){U?pe&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,0,0,me[ee].width,me[ee].height,ce,ue,me[ee].data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,je,me[ee].width,me[ee].height,0,ce,ue,me[ee].data);for(let Ie=0;Ie<he.length;Ie++){const mt=he[Ie].image[ee].image;U?pe&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie+1,0,0,mt.width,mt.height,ce,ue,mt.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie+1,je,mt.width,mt.height,0,ce,ue,mt.data)}}else{U?pe&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,0,0,ce,ue,me[ee]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,je,ce,ue,me[ee]);for(let Ie=0;Ie<he.length;Ie++){const We=he[Ie];U?pe&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie+1,0,0,ce,ue,We.image[ee]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ee,Ie+1,je,ce,ue,We.image[ee])}}}v(S)&&d(t.TEXTURE_CUBE_MAP),J.__version=re.version,S.onUpdate&&S.onUpdate(S)}R.__version=S.version}function ne(R,S,k,te,re,J){const be=s.convert(k.format,k.colorSpace),ge=s.convert(k.type),Ue=_(k.internalFormat,be,ge,k.colorSpace),Be=i.get(S),de=i.get(k);if(de.__renderTarget=S,!Be.__hasExternalTextures){const me=Math.max(1,S.width>>J),N=Math.max(1,S.height>>J);re===t.TEXTURE_3D||re===t.TEXTURE_2D_ARRAY?n.texImage3D(re,J,Ue,me,N,S.depth,0,be,ge,null):n.texImage2D(re,J,Ue,me,N,0,be,ge,null)}n.bindFramebuffer(t.FRAMEBUFFER,R),Re(S)?a.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,te,re,de.__webglTexture,0,L(S)):(re===t.TEXTURE_2D||re>=t.TEXTURE_CUBE_MAP_POSITIVE_X&&re<=t.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&t.framebufferTexture2D(t.FRAMEBUFFER,te,re,de.__webglTexture,J),n.bindFramebuffer(t.FRAMEBUFFER,null)}function oe(R,S,k){if(t.bindRenderbuffer(t.RENDERBUFFER,R),S.depthBuffer){const te=S.depthTexture,re=te&&te.isDepthTexture?te.type:null,J=b(S.stencilBuffer,re),be=S.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;Re(S)?a.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,L(S),J,S.width,S.height):k?t.renderbufferStorageMultisample(t.RENDERBUFFER,L(S),J,S.width,S.height):t.renderbufferStorage(t.RENDERBUFFER,J,S.width,S.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,be,t.RENDERBUFFER,R)}else{const te=S.textures;for(let re=0;re<te.length;re++){const J=te[re],be=s.convert(J.format,J.colorSpace),ge=s.convert(J.type),Ue=_(J.internalFormat,be,ge,J.colorSpace);Re(S)?a.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,L(S),Ue,S.width,S.height):k?t.renderbufferStorageMultisample(t.RENDERBUFFER,L(S),Ue,S.width,S.height):t.renderbufferStorage(t.RENDERBUFFER,Ue,S.width,S.height)}}t.bindRenderbuffer(t.RENDERBUFFER,null)}function ye(R,S,k){const te=S.isWebGLCubeRenderTarget===!0;if(n.bindFramebuffer(t.FRAMEBUFFER,R),!(S.depthTexture&&S.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const re=i.get(S.depthTexture);if(re.__renderTarget=S,(!re.__webglTexture||S.depthTexture.image.width!==S.width||S.depthTexture.image.height!==S.height)&&(S.depthTexture.image.width=S.width,S.depthTexture.image.height=S.height,S.depthTexture.needsUpdate=!0),te){if(re.__webglInit===void 0&&(re.__webglInit=!0,S.depthTexture.addEventListener("dispose",A)),re.__webglTexture===void 0){re.__webglTexture=t.createTexture(),n.bindTexture(t.TEXTURE_CUBE_MAP,re.__webglTexture),ae(t.TEXTURE_CUBE_MAP,S.depthTexture);const Be=s.convert(S.depthTexture.format),de=s.convert(S.depthTexture.type);let me;S.depthTexture.format===Ui?me=t.DEPTH_COMPONENT24:S.depthTexture.format===kr&&(me=t.DEPTH24_STENCIL8);for(let N=0;N<6;N++)t.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+N,0,me,S.width,S.height,0,Be,de,null)}}else V(S.depthTexture,0);const J=re.__webglTexture,be=L(S),ge=te?t.TEXTURE_CUBE_MAP_POSITIVE_X+k:t.TEXTURE_2D,Ue=S.depthTexture.format===kr?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;if(S.depthTexture.format===Ui)Re(S)?a.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,Ue,ge,J,0,be):t.framebufferTexture2D(t.FRAMEBUFFER,Ue,ge,J,0);else if(S.depthTexture.format===kr)Re(S)?a.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,Ue,ge,J,0,be):t.framebufferTexture2D(t.FRAMEBUFFER,Ue,ge,J,0);else throw new Error("Unknown depthTexture format")}function Le(R){const S=i.get(R),k=R.isWebGLCubeRenderTarget===!0;if(S.__boundDepthTexture!==R.depthTexture){const te=R.depthTexture;if(S.__depthDisposeCallback&&S.__depthDisposeCallback(),te){const re=()=>{delete S.__boundDepthTexture,delete S.__depthDisposeCallback,te.removeEventListener("dispose",re)};te.addEventListener("dispose",re),S.__depthDisposeCallback=re}S.__boundDepthTexture=te}if(R.depthTexture&&!S.__autoAllocateDepthBuffer)if(k)for(let te=0;te<6;te++)ye(S.__webglFramebuffer[te],R,te);else{const te=R.texture.mipmaps;te&&te.length>0?ye(S.__webglFramebuffer[0],R,0):ye(S.__webglFramebuffer,R,0)}else if(k){S.__webglDepthbuffer=[];for(let te=0;te<6;te++)if(n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer[te]),S.__webglDepthbuffer[te]===void 0)S.__webglDepthbuffer[te]=t.createRenderbuffer(),oe(S.__webglDepthbuffer[te],R,!1);else{const re=R.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,J=S.__webglDepthbuffer[te];t.bindRenderbuffer(t.RENDERBUFFER,J),t.framebufferRenderbuffer(t.FRAMEBUFFER,re,t.RENDERBUFFER,J)}}else{const te=R.texture.mipmaps;if(te&&te.length>0?n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer[0]):n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer),S.__webglDepthbuffer===void 0)S.__webglDepthbuffer=t.createRenderbuffer(),oe(S.__webglDepthbuffer,R,!1);else{const re=R.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,J=S.__webglDepthbuffer;t.bindRenderbuffer(t.RENDERBUFFER,J),t.framebufferRenderbuffer(t.FRAMEBUFFER,re,t.RENDERBUFFER,J)}}n.bindFramebuffer(t.FRAMEBUFFER,null)}function ht(R,S,k){const te=i.get(R);S!==void 0&&ne(te.__webglFramebuffer,R,R.texture,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,0),k!==void 0&&Le(R)}function ve(R){const S=R.texture,k=i.get(R),te=i.get(S);R.addEventListener("dispose",E);const re=R.textures,J=R.isWebGLCubeRenderTarget===!0,be=re.length>1;if(be||(te.__webglTexture===void 0&&(te.__webglTexture=t.createTexture()),te.__version=S.version,o.memory.textures++),J){k.__webglFramebuffer=[];for(let ge=0;ge<6;ge++)if(S.mipmaps&&S.mipmaps.length>0){k.__webglFramebuffer[ge]=[];for(let Ue=0;Ue<S.mipmaps.length;Ue++)k.__webglFramebuffer[ge][Ue]=t.createFramebuffer()}else k.__webglFramebuffer[ge]=t.createFramebuffer()}else{if(S.mipmaps&&S.mipmaps.length>0){k.__webglFramebuffer=[];for(let ge=0;ge<S.mipmaps.length;ge++)k.__webglFramebuffer[ge]=t.createFramebuffer()}else k.__webglFramebuffer=t.createFramebuffer();if(be)for(let ge=0,Ue=re.length;ge<Ue;ge++){const Be=i.get(re[ge]);Be.__webglTexture===void 0&&(Be.__webglTexture=t.createTexture(),o.memory.textures++)}if(R.samples>0&&Re(R)===!1){k.__webglMultisampledFramebuffer=t.createFramebuffer(),k.__webglColorRenderbuffer=[],n.bindFramebuffer(t.FRAMEBUFFER,k.__webglMultisampledFramebuffer);for(let ge=0;ge<re.length;ge++){const Ue=re[ge];k.__webglColorRenderbuffer[ge]=t.createRenderbuffer(),t.bindRenderbuffer(t.RENDERBUFFER,k.__webglColorRenderbuffer[ge]);const Be=s.convert(Ue.format,Ue.colorSpace),de=s.convert(Ue.type),me=_(Ue.internalFormat,Be,de,Ue.colorSpace,R.isXRRenderTarget===!0),N=L(R);t.renderbufferStorageMultisample(t.RENDERBUFFER,N,me,R.width,R.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+ge,t.RENDERBUFFER,k.__webglColorRenderbuffer[ge])}t.bindRenderbuffer(t.RENDERBUFFER,null),R.depthBuffer&&(k.__webglDepthRenderbuffer=t.createRenderbuffer(),oe(k.__webglDepthRenderbuffer,R,!0)),n.bindFramebuffer(t.FRAMEBUFFER,null)}}if(J){n.bindTexture(t.TEXTURE_CUBE_MAP,te.__webglTexture),ae(t.TEXTURE_CUBE_MAP,S);for(let ge=0;ge<6;ge++)if(S.mipmaps&&S.mipmaps.length>0)for(let Ue=0;Ue<S.mipmaps.length;Ue++)ne(k.__webglFramebuffer[ge][Ue],R,S,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+ge,Ue);else ne(k.__webglFramebuffer[ge],R,S,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0);v(S)&&d(t.TEXTURE_CUBE_MAP),n.unbindTexture()}else if(be){for(let ge=0,Ue=re.length;ge<Ue;ge++){const Be=re[ge],de=i.get(Be);let me=t.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(me=R.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(me,de.__webglTexture),ae(me,Be),ne(k.__webglFramebuffer,R,Be,t.COLOR_ATTACHMENT0+ge,me,0),v(Be)&&d(me)}n.unbindTexture()}else{let ge=t.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(ge=R.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(ge,te.__webglTexture),ae(ge,S),S.mipmaps&&S.mipmaps.length>0)for(let Ue=0;Ue<S.mipmaps.length;Ue++)ne(k.__webglFramebuffer[Ue],R,S,t.COLOR_ATTACHMENT0,ge,Ue);else ne(k.__webglFramebuffer,R,S,t.COLOR_ATTACHMENT0,ge,0);v(S)&&d(ge),n.unbindTexture()}R.depthBuffer&&Le(R)}function Ve(R){const S=R.textures;for(let k=0,te=S.length;k<te;k++){const re=S[k];if(v(re)){const J=m(R),be=i.get(re).__webglTexture;n.bindTexture(J,be),d(J),n.unbindTexture()}}}const ie=[],le=[];function ze(R){if(R.samples>0){if(Re(R)===!1){const S=R.textures,k=R.width,te=R.height;let re=t.COLOR_BUFFER_BIT;const J=R.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,be=i.get(R),ge=S.length>1;if(ge)for(let Be=0;Be<S.length;Be++)n.bindFramebuffer(t.FRAMEBUFFER,be.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+Be,t.RENDERBUFFER,null),n.bindFramebuffer(t.FRAMEBUFFER,be.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+Be,t.TEXTURE_2D,null,0);n.bindFramebuffer(t.READ_FRAMEBUFFER,be.__webglMultisampledFramebuffer);const Ue=R.texture.mipmaps;Ue&&Ue.length>0?n.bindFramebuffer(t.DRAW_FRAMEBUFFER,be.__webglFramebuffer[0]):n.bindFramebuffer(t.DRAW_FRAMEBUFFER,be.__webglFramebuffer);for(let Be=0;Be<S.length;Be++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(re|=t.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(re|=t.STENCIL_BUFFER_BIT)),ge){t.framebufferRenderbuffer(t.READ_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.RENDERBUFFER,be.__webglColorRenderbuffer[Be]);const de=i.get(S[Be]).__webglTexture;t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,de,0)}t.blitFramebuffer(0,0,k,te,0,0,k,te,re,t.NEAREST),c===!0&&(ie.length=0,le.length=0,ie.push(t.COLOR_ATTACHMENT0+Be),R.depthBuffer&&R.resolveDepthBuffer===!1&&(ie.push(J),le.push(J),t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,le)),t.invalidateFramebuffer(t.READ_FRAMEBUFFER,ie))}if(n.bindFramebuffer(t.READ_FRAMEBUFFER,null),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),ge)for(let Be=0;Be<S.length;Be++){n.bindFramebuffer(t.FRAMEBUFFER,be.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+Be,t.RENDERBUFFER,be.__webglColorRenderbuffer[Be]);const de=i.get(S[Be]).__webglTexture;n.bindFramebuffer(t.FRAMEBUFFER,be.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+Be,t.TEXTURE_2D,de,0)}n.bindFramebuffer(t.DRAW_FRAMEBUFFER,be.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&c){const S=R.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,[S])}}}function L(R){return Math.min(r.maxSamples,R.samples)}function Re(R){const S=i.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&S.__useRenderToTexture!==!1}function Ge(R){const S=o.render.frame;p.get(R)!==S&&(p.set(R,S),R.update())}function $e(R,S){const k=R.colorSpace,te=R.format,re=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||k!==$s&&k!==Zi&&(rt.getTransfer(k)===lt?(te!==Xn||re!==bn)&&He("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):it("WebGLTextures: Unsupported texture color space:",k)),S}function Te(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(u.width=R.naturalWidth||R.width,u.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(u.width=R.displayWidth,u.height=R.displayHeight):(u.width=R.width,u.height=R.height),u}this.allocateTextureUnit=B,this.resetTextureUnits=F,this.setTexture2D=V,this.setTexture2DArray=G,this.setTexture3D=z,this.setTextureCube=j,this.rebindTextures=ht,this.setupRenderTarget=ve,this.updateRenderTargetMipmap=Ve,this.updateMultisampleRenderTarget=ze,this.setupDepthRenderbuffer=Le,this.setupFrameBufferTexture=ne,this.useMultisampledRTT=Re,this.isReversedDepthBuffer=function(){return n.buffers.depth.getReversed()}}function wT(t,e){function n(i,r=Zi){let s;const o=rt.getTransfer(r);if(i===bn)return t.UNSIGNED_BYTE;if(i===kp)return t.UNSIGNED_SHORT_4_4_4_4;if(i===Op)return t.UNSIGNED_SHORT_5_5_5_1;if(i===$1)return t.UNSIGNED_INT_5_9_9_9_REV;if(i===K1)return t.UNSIGNED_INT_10F_11F_11F_REV;if(i===X1)return t.BYTE;if(i===q1)return t.SHORT;if(i===ea)return t.UNSIGNED_SHORT;if(i===Up)return t.INT;if(i===fi)return t.UNSIGNED_INT;if(i===oi)return t.FLOAT;if(i===Ni)return t.HALF_FLOAT;if(i===Y1)return t.ALPHA;if(i===Z1)return t.RGB;if(i===Xn)return t.RGBA;if(i===Ui)return t.DEPTH_COMPONENT;if(i===kr)return t.DEPTH_STENCIL;if(i===J1)return t.RED;if(i===zp)return t.RED_INTEGER;if(i===qs)return t.RG;if(i===Bp)return t.RG_INTEGER;if(i===jp)return t.RGBA_INTEGER;if(i===Cl||i===Al||i===Rl||i===Il)if(o===lt)if(s=e.get("WEBGL_compressed_texture_s3tc_srgb"),s!==null){if(i===Cl)return s.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===Al)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===Rl)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===Il)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(s=e.get("WEBGL_compressed_texture_s3tc"),s!==null){if(i===Cl)return s.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===Al)return s.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===Rl)return s.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===Il)return s.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===Jd||i===Qd||i===ef||i===tf)if(s=e.get("WEBGL_compressed_texture_pvrtc"),s!==null){if(i===Jd)return s.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===Qd)return s.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===ef)return s.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===tf)return s.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===nf||i===rf||i===sf||i===of||i===af||i===lf||i===cf)if(s=e.get("WEBGL_compressed_texture_etc"),s!==null){if(i===nf||i===rf)return o===lt?s.COMPRESSED_SRGB8_ETC2:s.COMPRESSED_RGB8_ETC2;if(i===sf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:s.COMPRESSED_RGBA8_ETC2_EAC;if(i===of)return s.COMPRESSED_R11_EAC;if(i===af)return s.COMPRESSED_SIGNED_R11_EAC;if(i===lf)return s.COMPRESSED_RG11_EAC;if(i===cf)return s.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===uf||i===df||i===ff||i===pf||i===hf||i===mf||i===gf||i===xf||i===vf||i===_f||i===yf||i===Sf||i===Mf||i===bf)if(s=e.get("WEBGL_compressed_texture_astc"),s!==null){if(i===uf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:s.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===df)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:s.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===ff)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:s.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===pf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:s.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===hf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:s.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===mf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:s.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===gf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:s.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===xf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:s.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===vf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:s.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===_f)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:s.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===yf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:s.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===Sf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:s.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===Mf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:s.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===bf)return o===lt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:s.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===Ef||i===Tf||i===wf)if(s=e.get("EXT_texture_compression_bptc"),s!==null){if(i===Ef)return o===lt?s.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:s.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===Tf)return s.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===wf)return s.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===Cf||i===Af||i===Rf||i===If)if(s=e.get("EXT_texture_compression_rgtc"),s!==null){if(i===Cf)return s.COMPRESSED_RED_RGTC1_EXT;if(i===Af)return s.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===Rf)return s.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===If)return s.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===ta?t.UNSIGNED_INT_24_8:t[i]!==void 0?t[i]:null}return{convert:n}}const CT=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,AT=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class RT{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,n){if(this.texture===null){const i=new dx(e.texture);(e.depthNear!==n.depthNear||e.depthFar!==n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){const n=e.cameras[0].viewport,i=new hi({vertexShader:CT,fragmentShader:AT,uniforms:{depthColor:{value:this.texture},depthWidth:{value:n.z},depthHeight:{value:n.w}}});this.mesh=new pn(new Ic(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class IT extends eo{constructor(e,n){super();const i=this;let r=null,s=1,o=null,a="local-floor",c=1,u=null,p=null,h=null,f=null,g=null,x=null;const M=typeof XRWebGLBinding<"u",v=new RT,d={},m=n.getContextAttributes();let _=null,b=null;const w=[],A=[],E=new Ze;let y=null;const C=new Mn;C.viewport=new wt;const P=new Mn;P.viewport=new wt;const I=[C,P],F=new jS;let B=null,W=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(D){let q=w[D];return q===void 0&&(q=new Su,w[D]=q),q.getTargetRaySpace()},this.getControllerGrip=function(D){let q=w[D];return q===void 0&&(q=new Su,w[D]=q),q.getGripSpace()},this.getHand=function(D){let q=w[D];return q===void 0&&(q=new Su,w[D]=q),q.getHandSpace()};function V(D){const q=A.indexOf(D.inputSource);if(q===-1)return;const ne=w[q];ne!==void 0&&(ne.update(D.inputSource,D.frame,u||o),ne.dispatchEvent({type:D.type,data:D.inputSource}))}function G(){r.removeEventListener("select",V),r.removeEventListener("selectstart",V),r.removeEventListener("selectend",V),r.removeEventListener("squeeze",V),r.removeEventListener("squeezestart",V),r.removeEventListener("squeezeend",V),r.removeEventListener("end",G),r.removeEventListener("inputsourceschange",z);for(let D=0;D<w.length;D++){const q=A[D];q!==null&&(A[D]=null,w[D].disconnect(q))}B=null,W=null,v.reset();for(const D in d)delete d[D];e.setRenderTarget(_),g=null,f=null,h=null,r=null,b=null,Oe.stop(),i.isPresenting=!1,e.setPixelRatio(y),e.setSize(E.width,E.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(D){s=D,i.isPresenting===!0&&He("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(D){a=D,i.isPresenting===!0&&He("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return u||o},this.setReferenceSpace=function(D){u=D},this.getBaseLayer=function(){return f!==null?f:g},this.getBinding=function(){return h===null&&M&&(h=new XRWebGLBinding(r,n)),h},this.getFrame=function(){return x},this.getSession=function(){return r},this.setSession=async function(D){if(r=D,r!==null){if(_=e.getRenderTarget(),r.addEventListener("select",V),r.addEventListener("selectstart",V),r.addEventListener("selectend",V),r.addEventListener("squeeze",V),r.addEventListener("squeezestart",V),r.addEventListener("squeezeend",V),r.addEventListener("end",G),r.addEventListener("inputsourceschange",z),m.xrCompatible!==!0&&await n.makeXRCompatible(),y=e.getPixelRatio(),e.getSize(E),M&&"createProjectionLayer"in XRWebGLBinding.prototype){let ne=null,oe=null,ye=null;m.depth&&(ye=m.stencil?n.DEPTH24_STENCIL8:n.DEPTH_COMPONENT24,ne=m.stencil?kr:Ui,oe=m.stencil?ta:fi);const Le={colorFormat:n.RGBA8,depthFormat:ye,scaleFactor:s};h=this.getBinding(),f=h.createProjectionLayer(Le),r.updateRenderState({layers:[f]}),e.setPixelRatio(1),e.setSize(f.textureWidth,f.textureHeight,!1),b=new di(f.textureWidth,f.textureHeight,{format:Xn,type:bn,depthTexture:new ia(f.textureWidth,f.textureHeight,oe,void 0,void 0,void 0,void 0,void 0,void 0,ne),stencilBuffer:m.stencil,colorSpace:e.outputColorSpace,samples:m.antialias?4:0,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}else{const ne={antialias:m.antialias,alpha:!0,depth:m.depth,stencil:m.stencil,framebufferScaleFactor:s};g=new XRWebGLLayer(r,n,ne),r.updateRenderState({baseLayer:g}),e.setPixelRatio(1),e.setSize(g.framebufferWidth,g.framebufferHeight,!1),b=new di(g.framebufferWidth,g.framebufferHeight,{format:Xn,type:bn,colorSpace:e.outputColorSpace,stencilBuffer:m.stencil,resolveDepthBuffer:g.ignoreDepthValues===!1,resolveStencilBuffer:g.ignoreDepthValues===!1})}b.isXRRenderTarget=!0,this.setFoveation(c),u=null,o=await r.requestReferenceSpace(a),Oe.setContext(r),Oe.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(r!==null)return r.environmentBlendMode},this.getDepthTexture=function(){return v.getDepthTexture()};function z(D){for(let q=0;q<D.removed.length;q++){const ne=D.removed[q],oe=A.indexOf(ne);oe>=0&&(A[oe]=null,w[oe].disconnect(ne))}for(let q=0;q<D.added.length;q++){const ne=D.added[q];let oe=A.indexOf(ne);if(oe===-1){for(let Le=0;Le<w.length;Le++)if(Le>=A.length){A.push(ne),oe=Le;break}else if(A[Le]===null){A[Le]=ne,oe=Le;break}if(oe===-1)break}const ye=w[oe];ye&&ye.connect(ne)}}const j=new H,$=new H;function Q(D,q,ne){j.setFromMatrixPosition(q.matrixWorld),$.setFromMatrixPosition(ne.matrixWorld);const oe=j.distanceTo($),ye=q.projectionMatrix.elements,Le=ne.projectionMatrix.elements,ht=ye[14]/(ye[10]-1),ve=ye[14]/(ye[10]+1),Ve=(ye[9]+1)/ye[5],ie=(ye[9]-1)/ye[5],le=(ye[8]-1)/ye[0],ze=(Le[8]+1)/Le[0],L=ht*le,Re=ht*ze,Ge=oe/(-le+ze),$e=Ge*-le;if(q.matrixWorld.decompose(D.position,D.quaternion,D.scale),D.translateX($e),D.translateZ(Ge),D.matrixWorld.compose(D.position,D.quaternion,D.scale),D.matrixWorldInverse.copy(D.matrixWorld).invert(),ye[10]===-1)D.projectionMatrix.copy(q.projectionMatrix),D.projectionMatrixInverse.copy(q.projectionMatrixInverse);else{const Te=ht+Ge,R=ve+Ge,S=L-$e,k=Re+(oe-$e),te=Ve*ve/R*Te,re=ie*ve/R*Te;D.projectionMatrix.makePerspective(S,k,te,re,Te,R),D.projectionMatrixInverse.copy(D.projectionMatrix).invert()}}function se(D,q){q===null?D.matrixWorld.copy(D.matrix):D.matrixWorld.multiplyMatrices(q.matrixWorld,D.matrix),D.matrixWorldInverse.copy(D.matrixWorld).invert()}this.updateCamera=function(D){if(r===null)return;let q=D.near,ne=D.far;v.texture!==null&&(v.depthNear>0&&(q=v.depthNear),v.depthFar>0&&(ne=v.depthFar)),F.near=P.near=C.near=q,F.far=P.far=C.far=ne,(B!==F.near||W!==F.far)&&(r.updateRenderState({depthNear:F.near,depthFar:F.far}),B=F.near,W=F.far),F.layers.mask=D.layers.mask|6,C.layers.mask=F.layers.mask&-5,P.layers.mask=F.layers.mask&-3;const oe=D.parent,ye=F.cameras;se(F,oe);for(let Le=0;Le<ye.length;Le++)se(ye[Le],oe);ye.length===2?Q(F,C,P):F.projectionMatrix.copy(C.projectionMatrix),ae(D,F,oe)};function ae(D,q,ne){ne===null?D.matrix.copy(q.matrixWorld):(D.matrix.copy(ne.matrixWorld),D.matrix.invert(),D.matrix.multiply(q.matrixWorld)),D.matrix.decompose(D.position,D.quaternion,D.scale),D.updateMatrixWorld(!0),D.projectionMatrix.copy(q.projectionMatrix),D.projectionMatrixInverse.copy(q.projectionMatrixInverse),D.isPerspectiveCamera&&(D.fov=Df*2*Math.atan(1/D.projectionMatrix.elements[5]),D.zoom=1)}this.getCamera=function(){return F},this.getFoveation=function(){if(!(f===null&&g===null))return c},this.setFoveation=function(D){c=D,f!==null&&(f.fixedFoveation=D),g!==null&&g.fixedFoveation!==void 0&&(g.fixedFoveation=D)},this.hasDepthSensing=function(){return v.texture!==null},this.getDepthSensingMesh=function(){return v.getMesh(F)},this.getCameraTexture=function(D){return d[D]};let Ae=null;function De(D,q){if(p=q.getViewerPose(u||o),x=q,p!==null){const ne=p.views;g!==null&&(e.setRenderTargetFramebuffer(b,g.framebuffer),e.setRenderTarget(b));let oe=!1;ne.length!==F.cameras.length&&(F.cameras.length=0,oe=!0);for(let ve=0;ve<ne.length;ve++){const Ve=ne[ve];let ie=null;if(g!==null)ie=g.getViewport(Ve);else{const ze=h.getViewSubImage(f,Ve);ie=ze.viewport,ve===0&&(e.setRenderTargetTextures(b,ze.colorTexture,ze.depthStencilTexture),e.setRenderTarget(b))}let le=I[ve];le===void 0&&(le=new Mn,le.layers.enable(ve),le.viewport=new wt,I[ve]=le),le.matrix.fromArray(Ve.transform.matrix),le.matrix.decompose(le.position,le.quaternion,le.scale),le.projectionMatrix.fromArray(Ve.projectionMatrix),le.projectionMatrixInverse.copy(le.projectionMatrix).invert(),le.viewport.set(ie.x,ie.y,ie.width,ie.height),ve===0&&(F.matrix.copy(le.matrix),F.matrix.decompose(F.position,F.quaternion,F.scale)),oe===!0&&F.cameras.push(le)}const ye=r.enabledFeatures;if(ye&&ye.includes("depth-sensing")&&r.depthUsage=="gpu-optimized"&&M){h=i.getBinding();const ve=h.getDepthInformation(ne[0]);ve&&ve.isValid&&ve.texture&&v.init(ve,r.renderState)}if(ye&&ye.includes("camera-access")&&M){e.state.unbindTexture(),h=i.getBinding();for(let ve=0;ve<ne.length;ve++){const Ve=ne[ve].camera;if(Ve){let ie=d[Ve];ie||(ie=new dx,d[Ve]=ie);const le=h.getCameraImage(Ve);ie.sourceTexture=le}}}}for(let ne=0;ne<w.length;ne++){const oe=A[ne],ye=w[ne];oe!==null&&ye!==void 0&&ye.update(oe,q,u||o)}Ae&&Ae(D,q),q.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:q}),x=null}const Oe=new gx;Oe.setAnimationLoop(De),this.setAnimationLoop=function(D){Ae=D},this.dispose=function(){}}}const wr=new pi,PT=new _t;function DT(t,e){function n(v,d){v.matrixAutoUpdate===!0&&v.updateMatrix(),d.value.copy(v.matrix)}function i(v,d){d.color.getRGB(v.fogColor.value,fx(t)),d.isFog?(v.fogNear.value=d.near,v.fogFar.value=d.far):d.isFogExp2&&(v.fogDensity.value=d.density)}function r(v,d,m,_,b){d.isMeshBasicMaterial?s(v,d):d.isMeshLambertMaterial?(s(v,d),d.envMap&&(v.envMapIntensity.value=d.envMapIntensity)):d.isMeshToonMaterial?(s(v,d),h(v,d)):d.isMeshPhongMaterial?(s(v,d),p(v,d),d.envMap&&(v.envMapIntensity.value=d.envMapIntensity)):d.isMeshStandardMaterial?(s(v,d),f(v,d),d.isMeshPhysicalMaterial&&g(v,d,b)):d.isMeshMatcapMaterial?(s(v,d),x(v,d)):d.isMeshDepthMaterial?s(v,d):d.isMeshDistanceMaterial?(s(v,d),M(v,d)):d.isMeshNormalMaterial?s(v,d):d.isLineBasicMaterial?(o(v,d),d.isLineDashedMaterial&&a(v,d)):d.isPointsMaterial?c(v,d,m,_):d.isSpriteMaterial?u(v,d):d.isShadowMaterial?(v.color.value.copy(d.color),v.opacity.value=d.opacity):d.isShaderMaterial&&(d.uniformsNeedUpdate=!1)}function s(v,d){v.opacity.value=d.opacity,d.color&&v.diffuse.value.copy(d.color),d.emissive&&v.emissive.value.copy(d.emissive).multiplyScalar(d.emissiveIntensity),d.map&&(v.map.value=d.map,n(d.map,v.mapTransform)),d.alphaMap&&(v.alphaMap.value=d.alphaMap,n(d.alphaMap,v.alphaMapTransform)),d.bumpMap&&(v.bumpMap.value=d.bumpMap,n(d.bumpMap,v.bumpMapTransform),v.bumpScale.value=d.bumpScale,d.side===en&&(v.bumpScale.value*=-1)),d.normalMap&&(v.normalMap.value=d.normalMap,n(d.normalMap,v.normalMapTransform),v.normalScale.value.copy(d.normalScale),d.side===en&&v.normalScale.value.negate()),d.displacementMap&&(v.displacementMap.value=d.displacementMap,n(d.displacementMap,v.displacementMapTransform),v.displacementScale.value=d.displacementScale,v.displacementBias.value=d.displacementBias),d.emissiveMap&&(v.emissiveMap.value=d.emissiveMap,n(d.emissiveMap,v.emissiveMapTransform)),d.specularMap&&(v.specularMap.value=d.specularMap,n(d.specularMap,v.specularMapTransform)),d.alphaTest>0&&(v.alphaTest.value=d.alphaTest);const m=e.get(d),_=m.envMap,b=m.envMapRotation;_&&(v.envMap.value=_,wr.copy(b),wr.x*=-1,wr.y*=-1,wr.z*=-1,_.isCubeTexture&&_.isRenderTargetTexture===!1&&(wr.y*=-1,wr.z*=-1),v.envMapRotation.value.setFromMatrix4(PT.makeRotationFromEuler(wr)),v.flipEnvMap.value=_.isCubeTexture&&_.isRenderTargetTexture===!1?-1:1,v.reflectivity.value=d.reflectivity,v.ior.value=d.ior,v.refractionRatio.value=d.refractionRatio),d.lightMap&&(v.lightMap.value=d.lightMap,v.lightMapIntensity.value=d.lightMapIntensity,n(d.lightMap,v.lightMapTransform)),d.aoMap&&(v.aoMap.value=d.aoMap,v.aoMapIntensity.value=d.aoMapIntensity,n(d.aoMap,v.aoMapTransform))}function o(v,d){v.diffuse.value.copy(d.color),v.opacity.value=d.opacity,d.map&&(v.map.value=d.map,n(d.map,v.mapTransform))}function a(v,d){v.dashSize.value=d.dashSize,v.totalSize.value=d.dashSize+d.gapSize,v.scale.value=d.scale}function c(v,d,m,_){v.diffuse.value.copy(d.color),v.opacity.value=d.opacity,v.size.value=d.size*m,v.scale.value=_*.5,d.map&&(v.map.value=d.map,n(d.map,v.uvTransform)),d.alphaMap&&(v.alphaMap.value=d.alphaMap,n(d.alphaMap,v.alphaMapTransform)),d.alphaTest>0&&(v.alphaTest.value=d.alphaTest)}function u(v,d){v.diffuse.value.copy(d.color),v.opacity.value=d.opacity,v.rotation.value=d.rotation,d.map&&(v.map.value=d.map,n(d.map,v.mapTransform)),d.alphaMap&&(v.alphaMap.value=d.alphaMap,n(d.alphaMap,v.alphaMapTransform)),d.alphaTest>0&&(v.alphaTest.value=d.alphaTest)}function p(v,d){v.specular.value.copy(d.specular),v.shininess.value=Math.max(d.shininess,1e-4)}function h(v,d){d.gradientMap&&(v.gradientMap.value=d.gradientMap)}function f(v,d){v.metalness.value=d.metalness,d.metalnessMap&&(v.metalnessMap.value=d.metalnessMap,n(d.metalnessMap,v.metalnessMapTransform)),v.roughness.value=d.roughness,d.roughnessMap&&(v.roughnessMap.value=d.roughnessMap,n(d.roughnessMap,v.roughnessMapTransform)),d.envMap&&(v.envMapIntensity.value=d.envMapIntensity)}function g(v,d,m){v.ior.value=d.ior,d.sheen>0&&(v.sheenColor.value.copy(d.sheenColor).multiplyScalar(d.sheen),v.sheenRoughness.value=d.sheenRoughness,d.sheenColorMap&&(v.sheenColorMap.value=d.sheenColorMap,n(d.sheenColorMap,v.sheenColorMapTransform)),d.sheenRoughnessMap&&(v.sheenRoughnessMap.value=d.sheenRoughnessMap,n(d.sheenRoughnessMap,v.sheenRoughnessMapTransform))),d.clearcoat>0&&(v.clearcoat.value=d.clearcoat,v.clearcoatRoughness.value=d.clearcoatRoughness,d.clearcoatMap&&(v.clearcoatMap.value=d.clearcoatMap,n(d.clearcoatMap,v.clearcoatMapTransform)),d.clearcoatRoughnessMap&&(v.clearcoatRoughnessMap.value=d.clearcoatRoughnessMap,n(d.clearcoatRoughnessMap,v.clearcoatRoughnessMapTransform)),d.clearcoatNormalMap&&(v.clearcoatNormalMap.value=d.clearcoatNormalMap,n(d.clearcoatNormalMap,v.clearcoatNormalMapTransform),v.clearcoatNormalScale.value.copy(d.clearcoatNormalScale),d.side===en&&v.clearcoatNormalScale.value.negate())),d.dispersion>0&&(v.dispersion.value=d.dispersion),d.iridescence>0&&(v.iridescence.value=d.iridescence,v.iridescenceIOR.value=d.iridescenceIOR,v.iridescenceThicknessMinimum.value=d.iridescenceThicknessRange[0],v.iridescenceThicknessMaximum.value=d.iridescenceThicknessRange[1],d.iridescenceMap&&(v.iridescenceMap.value=d.iridescenceMap,n(d.iridescenceMap,v.iridescenceMapTransform)),d.iridescenceThicknessMap&&(v.iridescenceThicknessMap.value=d.iridescenceThicknessMap,n(d.iridescenceThicknessMap,v.iridescenceThicknessMapTransform))),d.transmission>0&&(v.transmission.value=d.transmission,v.transmissionSamplerMap.value=m.texture,v.transmissionSamplerSize.value.set(m.width,m.height),d.transmissionMap&&(v.transmissionMap.value=d.transmissionMap,n(d.transmissionMap,v.transmissionMapTransform)),v.thickness.value=d.thickness,d.thicknessMap&&(v.thicknessMap.value=d.thicknessMap,n(d.thicknessMap,v.thicknessMapTransform)),v.attenuationDistance.value=d.attenuationDistance,v.attenuationColor.value.copy(d.attenuationColor)),d.anisotropy>0&&(v.anisotropyVector.value.set(d.anisotropy*Math.cos(d.anisotropyRotation),d.anisotropy*Math.sin(d.anisotropyRotation)),d.anisotropyMap&&(v.anisotropyMap.value=d.anisotropyMap,n(d.anisotropyMap,v.anisotropyMapTransform))),v.specularIntensity.value=d.specularIntensity,v.specularColor.value.copy(d.specularColor),d.specularColorMap&&(v.specularColorMap.value=d.specularColorMap,n(d.specularColorMap,v.specularColorMapTransform)),d.specularIntensityMap&&(v.specularIntensityMap.value=d.specularIntensityMap,n(d.specularIntensityMap,v.specularIntensityMapTransform))}function x(v,d){d.matcap&&(v.matcap.value=d.matcap)}function M(v,d){const m=e.get(d).light;v.referencePosition.value.setFromMatrixPosition(m.matrixWorld),v.nearDistance.value=m.shadow.camera.near,v.farDistance.value=m.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:r}}function LT(t,e,n,i){let r={},s={},o=[];const a=t.getParameter(t.MAX_UNIFORM_BUFFER_BINDINGS);function c(m,_){const b=_.program;i.uniformBlockBinding(m,b)}function u(m,_){let b=r[m.id];b===void 0&&(x(m),b=p(m),r[m.id]=b,m.addEventListener("dispose",v));const w=_.program;i.updateUBOMapping(m,w);const A=e.render.frame;s[m.id]!==A&&(f(m),s[m.id]=A)}function p(m){const _=h();m.__bindingPointIndex=_;const b=t.createBuffer(),w=m.__size,A=m.usage;return t.bindBuffer(t.UNIFORM_BUFFER,b),t.bufferData(t.UNIFORM_BUFFER,w,A),t.bindBuffer(t.UNIFORM_BUFFER,null),t.bindBufferBase(t.UNIFORM_BUFFER,_,b),b}function h(){for(let m=0;m<a;m++)if(o.indexOf(m)===-1)return o.push(m),m;return it("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function f(m){const _=r[m.id],b=m.uniforms,w=m.__cache;t.bindBuffer(t.UNIFORM_BUFFER,_);for(let A=0,E=b.length;A<E;A++){const y=Array.isArray(b[A])?b[A]:[b[A]];for(let C=0,P=y.length;C<P;C++){const I=y[C];if(g(I,A,C,w)===!0){const F=I.__offset,B=Array.isArray(I.value)?I.value:[I.value];let W=0;for(let V=0;V<B.length;V++){const G=B[V],z=M(G);typeof G=="number"||typeof G=="boolean"?(I.__data[0]=G,t.bufferSubData(t.UNIFORM_BUFFER,F+W,I.__data)):G.isMatrix3?(I.__data[0]=G.elements[0],I.__data[1]=G.elements[1],I.__data[2]=G.elements[2],I.__data[3]=0,I.__data[4]=G.elements[3],I.__data[5]=G.elements[4],I.__data[6]=G.elements[5],I.__data[7]=0,I.__data[8]=G.elements[6],I.__data[9]=G.elements[7],I.__data[10]=G.elements[8],I.__data[11]=0):(G.toArray(I.__data,W),W+=z.storage/Float32Array.BYTES_PER_ELEMENT)}t.bufferSubData(t.UNIFORM_BUFFER,F,I.__data)}}}t.bindBuffer(t.UNIFORM_BUFFER,null)}function g(m,_,b,w){const A=m.value,E=_+"_"+b;if(w[E]===void 0)return typeof A=="number"||typeof A=="boolean"?w[E]=A:w[E]=A.clone(),!0;{const y=w[E];if(typeof A=="number"||typeof A=="boolean"){if(y!==A)return w[E]=A,!0}else if(y.equals(A)===!1)return y.copy(A),!0}return!1}function x(m){const _=m.uniforms;let b=0;const w=16;for(let E=0,y=_.length;E<y;E++){const C=Array.isArray(_[E])?_[E]:[_[E]];for(let P=0,I=C.length;P<I;P++){const F=C[P],B=Array.isArray(F.value)?F.value:[F.value];for(let W=0,V=B.length;W<V;W++){const G=B[W],z=M(G),j=b%w,$=j%z.boundary,Q=j+$;b+=$,Q!==0&&w-Q<z.storage&&(b+=w-Q),F.__data=new Float32Array(z.storage/Float32Array.BYTES_PER_ELEMENT),F.__offset=b,b+=z.storage}}}const A=b%w;return A>0&&(b+=w-A),m.__size=b,m.__cache={},this}function M(m){const _={boundary:0,storage:0};return typeof m=="number"||typeof m=="boolean"?(_.boundary=4,_.storage=4):m.isVector2?(_.boundary=8,_.storage=8):m.isVector3||m.isColor?(_.boundary=16,_.storage=12):m.isVector4?(_.boundary=16,_.storage=16):m.isMatrix3?(_.boundary=48,_.storage=48):m.isMatrix4?(_.boundary=64,_.storage=64):m.isTexture?He("WebGLRenderer: Texture samplers can not be part of an uniforms group."):He("WebGLRenderer: Unsupported uniform value type.",m),_}function v(m){const _=m.target;_.removeEventListener("dispose",v);const b=o.indexOf(_.__bindingPointIndex);o.splice(b,1),t.deleteBuffer(r[_.id]),delete r[_.id],delete s[_.id]}function d(){for(const m in r)t.deleteBuffer(r[m]);o=[],r={},s={}}return{bind:c,update:u,dispose:d}}const FT=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]);let ei=null;function NT(){return ei===null&&(ei=new yS(FT,16,16,qs,Ni),ei.name="DFG_LUT",ei.minFilter=Qt,ei.magFilter=Qt,ei.wrapS=Ci,ei.wrapT=Ci,ei.generateMipmaps=!1,ei.needsUpdate=!0),ei}class UT{constructor(e={}){const{canvas:n=Ky(),context:i=null,depth:r=!0,stencil:s=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:u=!1,powerPreference:p="default",failIfMajorPerformanceCaveat:h=!1,reversedDepthBuffer:f=!1,outputBufferType:g=bn}=e;this.isWebGLRenderer=!0;let x;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");x=i.getContextAttributes().alpha}else x=o;const M=g,v=new Set([jp,Bp,zp]),d=new Set([bn,fi,ea,ta,kp,Op]),m=new Uint32Array(4),_=new Int32Array(4);let b=null,w=null;const A=[],E=[];let y=null;this.domElement=n,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=ui,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const C=this;let P=!1;this._outputColorSpace=In;let I=0,F=0,B=null,W=-1,V=null;const G=new wt,z=new wt;let j=null;const $=new tt(0);let Q=0,se=n.width,ae=n.height,Ae=1,De=null,Oe=null;const D=new wt(0,0,se,ae),q=new wt(0,0,se,ae);let ne=!1;const oe=new Xp;let ye=!1,Le=!1;const ht=new _t,ve=new H,Ve=new wt,ie={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let le=!1;function ze(){return B===null?Ae:1}let L=i;function Re(T,O){return n.getContext(T,O)}try{const T={alpha:!0,depth:r,stencil:s,antialias:a,premultipliedAlpha:c,preserveDrawingBuffer:u,powerPreference:p,failIfMajorPerformanceCaveat:h};if("setAttribute"in n&&n.setAttribute("data-engine",`three.js r${Np}`),n.addEventListener("webglcontextlost",Ie,!1),n.addEventListener("webglcontextrestored",We,!1),n.addEventListener("webglcontextcreationerror",mt,!1),L===null){const O="webgl2";if(L=Re(O,T),L===null)throw Re(O)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(T){throw it("WebGLRenderer: "+T.message),T}let Ge,$e,Te,R,S,k,te,re,J,be,ge,Ue,Be,de,me,N,ce,ue,je,U,xe,pe,Ee;function he(){Ge=new U2(L),Ge.init(),xe=new wT(L,Ge),$e=new A2(L,Ge,e,xe),Te=new ET(L,Ge),$e.reversedDepthBuffer&&f&&Te.buffers.depth.setReversed(!0),R=new z2(L),S=new uT,k=new TT(L,Ge,Te,S,$e,xe,R),te=new N2(C),re=new GS(L),pe=new w2(L,re),J=new k2(L,re,R,pe),be=new j2(L,J,re,pe,R),ue=new B2(L,$e,k),me=new R2(S),ge=new cT(C,te,Ge,$e,pe,me),Ue=new DT(C,S),Be=new fT,de=new vT(Ge),ce=new T2(C,te,Te,be,x,c),N=new bT(C,be,$e),Ee=new LT(L,R,$e,Te),je=new C2(L,Ge,R),U=new O2(L,Ge,R),R.programs=ge.programs,C.capabilities=$e,C.extensions=Ge,C.properties=S,C.renderLists=Be,C.shadowMap=N,C.state=Te,C.info=R}he(),M!==bn&&(y=new H2(M,n.width,n.height,r,s));const ee=new IT(C,L);this.xr=ee,this.getContext=function(){return L},this.getContextAttributes=function(){return L.getContextAttributes()},this.forceContextLoss=function(){const T=Ge.get("WEBGL_lose_context");T&&T.loseContext()},this.forceContextRestore=function(){const T=Ge.get("WEBGL_lose_context");T&&T.restoreContext()},this.getPixelRatio=function(){return Ae},this.setPixelRatio=function(T){T!==void 0&&(Ae=T,this.setSize(se,ae,!1))},this.getSize=function(T){return T.set(se,ae)},this.setSize=function(T,O,Y=!0){if(ee.isPresenting){He("WebGLRenderer: Can't change size while VR device is presenting.");return}se=T,ae=O,n.width=Math.floor(T*Ae),n.height=Math.floor(O*Ae),Y===!0&&(n.style.width=T+"px",n.style.height=O+"px"),y!==null&&y.setSize(n.width,n.height),this.setViewport(0,0,T,O)},this.getDrawingBufferSize=function(T){return T.set(se*Ae,ae*Ae).floor()},this.setDrawingBufferSize=function(T,O,Y){se=T,ae=O,Ae=Y,n.width=Math.floor(T*Y),n.height=Math.floor(O*Y),this.setViewport(0,0,T,O)},this.setEffects=function(T){if(M===bn){console.error("THREE.WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(T){for(let O=0;O<T.length;O++)if(T[O].isOutputPass===!0){console.warn("THREE.WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}y.setEffects(T||[])},this.getCurrentViewport=function(T){return T.copy(G)},this.getViewport=function(T){return T.copy(D)},this.setViewport=function(T,O,Y,K){T.isVector4?D.set(T.x,T.y,T.z,T.w):D.set(T,O,Y,K),Te.viewport(G.copy(D).multiplyScalar(Ae).round())},this.getScissor=function(T){return T.copy(q)},this.setScissor=function(T,O,Y,K){T.isVector4?q.set(T.x,T.y,T.z,T.w):q.set(T,O,Y,K),Te.scissor(z.copy(q).multiplyScalar(Ae).round())},this.getScissorTest=function(){return ne},this.setScissorTest=function(T){Te.setScissorTest(ne=T)},this.setOpaqueSort=function(T){De=T},this.setTransparentSort=function(T){Oe=T},this.getClearColor=function(T){return T.copy(ce.getClearColor())},this.setClearColor=function(){ce.setClearColor(...arguments)},this.getClearAlpha=function(){return ce.getClearAlpha()},this.setClearAlpha=function(){ce.setClearAlpha(...arguments)},this.clear=function(T=!0,O=!0,Y=!0){let K=0;if(T){let X=!1;if(B!==null){const Se=B.texture.format;X=v.has(Se)}if(X){const Se=B.texture.type,we=d.has(Se),Me=ce.getClearColor(),Pe=ce.getClearAlpha(),Ne=Me.r,Xe=Me.g,Ye=Me.b;we?(m[0]=Ne,m[1]=Xe,m[2]=Ye,m[3]=Pe,L.clearBufferuiv(L.COLOR,0,m)):(_[0]=Ne,_[1]=Xe,_[2]=Ye,_[3]=Pe,L.clearBufferiv(L.COLOR,0,_))}else K|=L.COLOR_BUFFER_BIT}O&&(K|=L.DEPTH_BUFFER_BIT),Y&&(K|=L.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),K!==0&&L.clear(K)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){n.removeEventListener("webglcontextlost",Ie,!1),n.removeEventListener("webglcontextrestored",We,!1),n.removeEventListener("webglcontextcreationerror",mt,!1),ce.dispose(),Be.dispose(),de.dispose(),S.dispose(),te.dispose(),be.dispose(),pe.dispose(),Ee.dispose(),ge.dispose(),ee.dispose(),ee.removeEventListener("sessionstart",nh),ee.removeEventListener("sessionend",ih),_r.stop()};function Ie(T){T.preventDefault(),oc("WebGLRenderer: Context Lost."),P=!0}function We(){oc("WebGLRenderer: Context Restored."),P=!1;const T=R.autoReset,O=N.enabled,Y=N.autoUpdate,K=N.needsUpdate,X=N.type;he(),R.autoReset=T,N.enabled=O,N.autoUpdate=Y,N.needsUpdate=K,N.type=X}function mt(T){it("WebGLRenderer: A WebGL context could not be created. Reason: ",T.statusMessage)}function at(T){const O=T.target;O.removeEventListener("dispose",at),gi(O)}function gi(T){xi(T),S.remove(T)}function xi(T){const O=S.get(T).programs;O!==void 0&&(O.forEach(function(Y){ge.releaseProgram(Y)}),T.isShaderMaterial&&ge.releaseShaderCache(T))}this.renderBufferDirect=function(T,O,Y,K,X,Se){O===null&&(O=ie);const we=X.isMesh&&X.matrixWorld.determinant()<0,Me=kx(T,O,Y,K,X);Te.setMaterial(K,we);let Pe=Y.index,Ne=1;if(K.wireframe===!0){if(Pe=J.getWireframeAttribute(Y),Pe===void 0)return;Ne=2}const Xe=Y.drawRange,Ye=Y.attributes.position;let ke=Xe.start*Ne,ut=(Xe.start+Xe.count)*Ne;Se!==null&&(ke=Math.max(ke,Se.start*Ne),ut=Math.min(ut,(Se.start+Se.count)*Ne)),Pe!==null?(ke=Math.max(ke,0),ut=Math.min(ut,Pe.count)):Ye!=null&&(ke=Math.max(ke,0),ut=Math.min(ut,Ye.count));const Ct=ut-ke;if(Ct<0||Ct===1/0)return;pe.setup(X,K,Me,Y,Pe);let Et,dt=je;if(Pe!==null&&(Et=re.get(Pe),dt=U,dt.setIndex(Et)),X.isMesh)K.wireframe===!0?(Te.setLineWidth(K.wireframeLinewidth*ze()),dt.setMode(L.LINES)):dt.setMode(L.TRIANGLES);else if(X.isLine){let qt=K.linewidth;qt===void 0&&(qt=1),Te.setLineWidth(qt*ze()),X.isLineSegments?dt.setMode(L.LINES):X.isLineLoop?dt.setMode(L.LINE_LOOP):dt.setMode(L.LINE_STRIP)}else X.isPoints?dt.setMode(L.POINTS):X.isSprite&&dt.setMode(L.TRIANGLES);if(X.isBatchedMesh)if(X._multiDrawInstances!==null)ac("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),dt.renderMultiDrawInstances(X._multiDrawStarts,X._multiDrawCounts,X._multiDrawCount,X._multiDrawInstances);else if(Ge.get("WEBGL_multi_draw"))dt.renderMultiDraw(X._multiDrawStarts,X._multiDrawCounts,X._multiDrawCount);else{const qt=X._multiDrawStarts,Fe=X._multiDrawCounts,vn=X._multiDrawCount,ot=Pe?re.get(Pe).bytesPerElement:1,On=S.get(K).currentProgram.getUniforms();for(let Zn=0;Zn<vn;Zn++)On.setValue(L,"_gl_DrawID",Zn),dt.render(qt[Zn]/ot,Fe[Zn])}else if(X.isInstancedMesh)dt.renderInstances(ke,Ct,X.count);else if(Y.isInstancedBufferGeometry){const qt=Y._maxInstanceCount!==void 0?Y._maxInstanceCount:1/0,Fe=Math.min(Y.instanceCount,qt);dt.renderInstances(ke,Ct,Fe)}else dt.render(ke,Ct)};function th(T,O,Y){T.transparent===!0&&T.side===ri&&T.forceSinglePass===!1?(T.side=en,T.needsUpdate=!0,_a(T,O,Y),T.side=pr,T.needsUpdate=!0,_a(T,O,Y),T.side=ri):_a(T,O,Y)}this.compile=function(T,O,Y=null){Y===null&&(Y=T),w=de.get(Y),w.init(O),E.push(w),Y.traverseVisible(function(X){X.isLight&&X.layers.test(O.layers)&&(w.pushLight(X),X.castShadow&&w.pushShadow(X))}),T!==Y&&T.traverseVisible(function(X){X.isLight&&X.layers.test(O.layers)&&(w.pushLight(X),X.castShadow&&w.pushShadow(X))}),w.setupLights();const K=new Set;return T.traverse(function(X){if(!(X.isMesh||X.isPoints||X.isLine||X.isSprite))return;const Se=X.material;if(Se)if(Array.isArray(Se))for(let we=0;we<Se.length;we++){const Me=Se[we];th(Me,Y,X),K.add(Me)}else th(Se,Y,X),K.add(Se)}),w=E.pop(),K},this.compileAsync=function(T,O,Y=null){const K=this.compile(T,O,Y);return new Promise(X=>{function Se(){if(K.forEach(function(we){S.get(we).currentProgram.isReady()&&K.delete(we)}),K.size===0){X(T);return}setTimeout(Se,10)}Ge.get("KHR_parallel_shader_compile")!==null?Se():setTimeout(Se,10)})};let kc=null;function Ux(T){kc&&kc(T)}function nh(){_r.stop()}function ih(){_r.start()}const _r=new gx;_r.setAnimationLoop(Ux),typeof self<"u"&&_r.setContext(self),this.setAnimationLoop=function(T){kc=T,ee.setAnimationLoop(T),T===null?_r.stop():_r.start()},ee.addEventListener("sessionstart",nh),ee.addEventListener("sessionend",ih),this.render=function(T,O){if(O!==void 0&&O.isCamera!==!0){it("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(P===!0)return;const Y=ee.enabled===!0&&ee.isPresenting===!0,K=y!==null&&(B===null||Y)&&y.begin(C,B);if(T.matrixWorldAutoUpdate===!0&&T.updateMatrixWorld(),O.parent===null&&O.matrixWorldAutoUpdate===!0&&O.updateMatrixWorld(),ee.enabled===!0&&ee.isPresenting===!0&&(y===null||y.isCompositing()===!1)&&(ee.cameraAutoUpdate===!0&&ee.updateCamera(O),O=ee.getCamera()),T.isScene===!0&&T.onBeforeRender(C,T,O,B),w=de.get(T,E.length),w.init(O),E.push(w),ht.multiplyMatrices(O.projectionMatrix,O.matrixWorldInverse),oe.setFromProjectionMatrix(ht,ai,O.reversedDepth),Le=this.localClippingEnabled,ye=me.init(this.clippingPlanes,Le),b=Be.get(T,A.length),b.init(),A.push(b),ee.enabled===!0&&ee.isPresenting===!0){const we=C.xr.getDepthSensingMesh();we!==null&&Oc(we,O,-1/0,C.sortObjects)}Oc(T,O,0,C.sortObjects),b.finish(),C.sortObjects===!0&&b.sort(De,Oe),le=ee.enabled===!1||ee.isPresenting===!1||ee.hasDepthSensing()===!1,le&&ce.addToRenderList(b,T),this.info.render.frame++,ye===!0&&me.beginShadows();const X=w.state.shadowsArray;if(N.render(X,T,O),ye===!0&&me.endShadows(),this.info.autoReset===!0&&this.info.reset(),(K&&y.hasRenderPass())===!1){const we=b.opaque,Me=b.transmissive;if(w.setupLights(),O.isArrayCamera){const Pe=O.cameras;if(Me.length>0)for(let Ne=0,Xe=Pe.length;Ne<Xe;Ne++){const Ye=Pe[Ne];sh(we,Me,T,Ye)}le&&ce.render(T);for(let Ne=0,Xe=Pe.length;Ne<Xe;Ne++){const Ye=Pe[Ne];rh(b,T,Ye,Ye.viewport)}}else Me.length>0&&sh(we,Me,T,O),le&&ce.render(T),rh(b,T,O)}B!==null&&F===0&&(k.updateMultisampleRenderTarget(B),k.updateRenderTargetMipmap(B)),K&&y.end(C),T.isScene===!0&&T.onAfterRender(C,T,O),pe.resetDefaultState(),W=-1,V=null,E.pop(),E.length>0?(w=E[E.length-1],ye===!0&&me.setGlobalState(C.clippingPlanes,w.state.camera)):w=null,A.pop(),A.length>0?b=A[A.length-1]:b=null};function Oc(T,O,Y,K){if(T.visible===!1)return;if(T.layers.test(O.layers)){if(T.isGroup)Y=T.renderOrder;else if(T.isLOD)T.autoUpdate===!0&&T.update(O);else if(T.isLight)w.pushLight(T),T.castShadow&&w.pushShadow(T);else if(T.isSprite){if(!T.frustumCulled||oe.intersectsSprite(T)){K&&Ve.setFromMatrixPosition(T.matrixWorld).applyMatrix4(ht);const we=be.update(T),Me=T.material;Me.visible&&b.push(T,we,Me,Y,Ve.z,null)}}else if((T.isMesh||T.isLine||T.isPoints)&&(!T.frustumCulled||oe.intersectsObject(T))){const we=be.update(T),Me=T.material;if(K&&(T.boundingSphere!==void 0?(T.boundingSphere===null&&T.computeBoundingSphere(),Ve.copy(T.boundingSphere.center)):(we.boundingSphere===null&&we.computeBoundingSphere(),Ve.copy(we.boundingSphere.center)),Ve.applyMatrix4(T.matrixWorld).applyMatrix4(ht)),Array.isArray(Me)){const Pe=we.groups;for(let Ne=0,Xe=Pe.length;Ne<Xe;Ne++){const Ye=Pe[Ne],ke=Me[Ye.materialIndex];ke&&ke.visible&&b.push(T,we,ke,Y,Ve.z,Ye)}}else Me.visible&&b.push(T,we,Me,Y,Ve.z,null)}}const Se=T.children;for(let we=0,Me=Se.length;we<Me;we++)Oc(Se[we],O,Y,K)}function rh(T,O,Y,K){const{opaque:X,transmissive:Se,transparent:we}=T;w.setupLightsView(Y),ye===!0&&me.setGlobalState(C.clippingPlanes,Y),K&&Te.viewport(G.copy(K)),X.length>0&&va(X,O,Y),Se.length>0&&va(Se,O,Y),we.length>0&&va(we,O,Y),Te.buffers.depth.setTest(!0),Te.buffers.depth.setMask(!0),Te.buffers.color.setMask(!0),Te.setPolygonOffset(!1)}function sh(T,O,Y,K){if((Y.isScene===!0?Y.overrideMaterial:null)!==null)return;if(w.state.transmissionRenderTarget[K.id]===void 0){const ke=Ge.has("EXT_color_buffer_half_float")||Ge.has("EXT_color_buffer_float");w.state.transmissionRenderTarget[K.id]=new di(1,1,{generateMipmaps:!0,type:ke?Ni:bn,minFilter:Ur,samples:Math.max(4,$e.samples),stencilBuffer:s,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:rt.workingColorSpace})}const Se=w.state.transmissionRenderTarget[K.id],we=K.viewport||G;Se.setSize(we.z*C.transmissionResolutionScale,we.w*C.transmissionResolutionScale);const Me=C.getRenderTarget(),Pe=C.getActiveCubeFace(),Ne=C.getActiveMipmapLevel();C.setRenderTarget(Se),C.getClearColor($),Q=C.getClearAlpha(),Q<1&&C.setClearColor(16777215,.5),C.clear(),le&&ce.render(Y);const Xe=C.toneMapping;C.toneMapping=ui;const Ye=K.viewport;if(K.viewport!==void 0&&(K.viewport=void 0),w.setupLightsView(K),ye===!0&&me.setGlobalState(C.clippingPlanes,K),va(T,Y,K),k.updateMultisampleRenderTarget(Se),k.updateRenderTargetMipmap(Se),Ge.has("WEBGL_multisampled_render_to_texture")===!1){let ke=!1;for(let ut=0,Ct=O.length;ut<Ct;ut++){const Et=O[ut],{object:dt,geometry:qt,material:Fe,group:vn}=Et;if(Fe.side===ri&&dt.layers.test(K.layers)){const ot=Fe.side;Fe.side=en,Fe.needsUpdate=!0,oh(dt,Y,K,qt,Fe,vn),Fe.side=ot,Fe.needsUpdate=!0,ke=!0}}ke===!0&&(k.updateMultisampleRenderTarget(Se),k.updateRenderTargetMipmap(Se))}C.setRenderTarget(Me,Pe,Ne),C.setClearColor($,Q),Ye!==void 0&&(K.viewport=Ye),C.toneMapping=Xe}function va(T,O,Y){const K=O.isScene===!0?O.overrideMaterial:null;for(let X=0,Se=T.length;X<Se;X++){const we=T[X],{object:Me,geometry:Pe,group:Ne}=we;let Xe=we.material;Xe.allowOverride===!0&&K!==null&&(Xe=K),Me.layers.test(Y.layers)&&oh(Me,O,Y,Pe,Xe,Ne)}}function oh(T,O,Y,K,X,Se){T.onBeforeRender(C,O,Y,K,X,Se),T.modelViewMatrix.multiplyMatrices(Y.matrixWorldInverse,T.matrixWorld),T.normalMatrix.getNormalMatrix(T.modelViewMatrix),X.onBeforeRender(C,O,Y,K,T,Se),X.transparent===!0&&X.side===ri&&X.forceSinglePass===!1?(X.side=en,X.needsUpdate=!0,C.renderBufferDirect(Y,O,K,X,T,Se),X.side=pr,X.needsUpdate=!0,C.renderBufferDirect(Y,O,K,X,T,Se),X.side=ri):C.renderBufferDirect(Y,O,K,X,T,Se),T.onAfterRender(C,O,Y,K,X,Se)}function _a(T,O,Y){O.isScene!==!0&&(O=ie);const K=S.get(T),X=w.state.lights,Se=w.state.shadowsArray,we=X.state.version,Me=ge.getParameters(T,X.state,Se,O,Y),Pe=ge.getProgramCacheKey(Me);let Ne=K.programs;K.environment=T.isMeshStandardMaterial||T.isMeshLambertMaterial||T.isMeshPhongMaterial?O.environment:null,K.fog=O.fog;const Xe=T.isMeshStandardMaterial||T.isMeshLambertMaterial&&!T.envMap||T.isMeshPhongMaterial&&!T.envMap;K.envMap=te.get(T.envMap||K.environment,Xe),K.envMapRotation=K.environment!==null&&T.envMap===null?O.environmentRotation:T.envMapRotation,Ne===void 0&&(T.addEventListener("dispose",at),Ne=new Map,K.programs=Ne);let Ye=Ne.get(Pe);if(Ye!==void 0){if(K.currentProgram===Ye&&K.lightsStateVersion===we)return lh(T,Me),Ye}else Me.uniforms=ge.getUniforms(T),T.onBeforeCompile(Me,C),Ye=ge.acquireProgram(Me,Pe),Ne.set(Pe,Ye),K.uniforms=Me.uniforms;const ke=K.uniforms;return(!T.isShaderMaterial&&!T.isRawShaderMaterial||T.clipping===!0)&&(ke.clippingPlanes=me.uniform),lh(T,Me),K.needsLights=zx(T),K.lightsStateVersion=we,K.needsLights&&(ke.ambientLightColor.value=X.state.ambient,ke.lightProbe.value=X.state.probe,ke.directionalLights.value=X.state.directional,ke.directionalLightShadows.value=X.state.directionalShadow,ke.spotLights.value=X.state.spot,ke.spotLightShadows.value=X.state.spotShadow,ke.rectAreaLights.value=X.state.rectArea,ke.ltc_1.value=X.state.rectAreaLTC1,ke.ltc_2.value=X.state.rectAreaLTC2,ke.pointLights.value=X.state.point,ke.pointLightShadows.value=X.state.pointShadow,ke.hemisphereLights.value=X.state.hemi,ke.directionalShadowMatrix.value=X.state.directionalShadowMatrix,ke.spotLightMatrix.value=X.state.spotLightMatrix,ke.spotLightMap.value=X.state.spotLightMap,ke.pointShadowMatrix.value=X.state.pointShadowMatrix),K.currentProgram=Ye,K.uniformsList=null,Ye}function ah(T){if(T.uniformsList===null){const O=T.currentProgram.getUniforms();T.uniformsList=Pl.seqWithValue(O.seq,T.uniforms)}return T.uniformsList}function lh(T,O){const Y=S.get(T);Y.outputColorSpace=O.outputColorSpace,Y.batching=O.batching,Y.batchingColor=O.batchingColor,Y.instancing=O.instancing,Y.instancingColor=O.instancingColor,Y.instancingMorph=O.instancingMorph,Y.skinning=O.skinning,Y.morphTargets=O.morphTargets,Y.morphNormals=O.morphNormals,Y.morphColors=O.morphColors,Y.morphTargetsCount=O.morphTargetsCount,Y.numClippingPlanes=O.numClippingPlanes,Y.numIntersection=O.numClipIntersection,Y.vertexAlphas=O.vertexAlphas,Y.vertexTangents=O.vertexTangents,Y.toneMapping=O.toneMapping}function kx(T,O,Y,K,X){O.isScene!==!0&&(O=ie),k.resetTextureUnits();const Se=O.fog,we=K.isMeshStandardMaterial||K.isMeshLambertMaterial||K.isMeshPhongMaterial?O.environment:null,Me=B===null?C.outputColorSpace:B.isXRRenderTarget===!0?B.texture.colorSpace:$s,Pe=K.isMeshStandardMaterial||K.isMeshLambertMaterial&&!K.envMap||K.isMeshPhongMaterial&&!K.envMap,Ne=te.get(K.envMap||we,Pe),Xe=K.vertexColors===!0&&!!Y.attributes.color&&Y.attributes.color.itemSize===4,Ye=!!Y.attributes.tangent&&(!!K.normalMap||K.anisotropy>0),ke=!!Y.morphAttributes.position,ut=!!Y.morphAttributes.normal,Ct=!!Y.morphAttributes.color;let Et=ui;K.toneMapped&&(B===null||B.isXRRenderTarget===!0)&&(Et=C.toneMapping);const dt=Y.morphAttributes.position||Y.morphAttributes.normal||Y.morphAttributes.color,qt=dt!==void 0?dt.length:0,Fe=S.get(K),vn=w.state.lights;if(ye===!0&&(Le===!0||T!==V)){const Nt=T===V&&K.id===W;me.setState(K,T,Nt)}let ot=!1;K.version===Fe.__version?(Fe.needsLights&&Fe.lightsStateVersion!==vn.state.version||Fe.outputColorSpace!==Me||X.isBatchedMesh&&Fe.batching===!1||!X.isBatchedMesh&&Fe.batching===!0||X.isBatchedMesh&&Fe.batchingColor===!0&&X.colorTexture===null||X.isBatchedMesh&&Fe.batchingColor===!1&&X.colorTexture!==null||X.isInstancedMesh&&Fe.instancing===!1||!X.isInstancedMesh&&Fe.instancing===!0||X.isSkinnedMesh&&Fe.skinning===!1||!X.isSkinnedMesh&&Fe.skinning===!0||X.isInstancedMesh&&Fe.instancingColor===!0&&X.instanceColor===null||X.isInstancedMesh&&Fe.instancingColor===!1&&X.instanceColor!==null||X.isInstancedMesh&&Fe.instancingMorph===!0&&X.morphTexture===null||X.isInstancedMesh&&Fe.instancingMorph===!1&&X.morphTexture!==null||Fe.envMap!==Ne||K.fog===!0&&Fe.fog!==Se||Fe.numClippingPlanes!==void 0&&(Fe.numClippingPlanes!==me.numPlanes||Fe.numIntersection!==me.numIntersection)||Fe.vertexAlphas!==Xe||Fe.vertexTangents!==Ye||Fe.morphTargets!==ke||Fe.morphNormals!==ut||Fe.morphColors!==Ct||Fe.toneMapping!==Et||Fe.morphTargetsCount!==qt)&&(ot=!0):(ot=!0,Fe.__version=K.version);let On=Fe.currentProgram;ot===!0&&(On=_a(K,O,X));let Zn=!1,yr=!1,Kr=!1;const pt=On.getUniforms(),zt=Fe.uniforms;if(Te.useProgram(On.program)&&(Zn=!0,yr=!0,Kr=!0),K.id!==W&&(W=K.id,yr=!0),Zn||V!==T){Te.buffers.depth.getReversed()&&T.reversedDepth!==!0&&(T._reversedDepth=!0,T.updateProjectionMatrix()),pt.setValue(L,"projectionMatrix",T.projectionMatrix),pt.setValue(L,"viewMatrix",T.matrixWorldInverse);const zi=pt.map.cameraPosition;zi!==void 0&&zi.setValue(L,ve.setFromMatrixPosition(T.matrixWorld)),$e.logarithmicDepthBuffer&&pt.setValue(L,"logDepthBufFC",2/(Math.log(T.far+1)/Math.LN2)),(K.isMeshPhongMaterial||K.isMeshToonMaterial||K.isMeshLambertMaterial||K.isMeshBasicMaterial||K.isMeshStandardMaterial||K.isShaderMaterial)&&pt.setValue(L,"isOrthographic",T.isOrthographicCamera===!0),V!==T&&(V=T,yr=!0,Kr=!0)}if(Fe.needsLights&&(vn.state.directionalShadowMap.length>0&&pt.setValue(L,"directionalShadowMap",vn.state.directionalShadowMap,k),vn.state.spotShadowMap.length>0&&pt.setValue(L,"spotShadowMap",vn.state.spotShadowMap,k),vn.state.pointShadowMap.length>0&&pt.setValue(L,"pointShadowMap",vn.state.pointShadowMap,k)),X.isSkinnedMesh){pt.setOptional(L,X,"bindMatrix"),pt.setOptional(L,X,"bindMatrixInverse");const Nt=X.skeleton;Nt&&(Nt.boneTexture===null&&Nt.computeBoneTexture(),pt.setValue(L,"boneTexture",Nt.boneTexture,k))}X.isBatchedMesh&&(pt.setOptional(L,X,"batchingTexture"),pt.setValue(L,"batchingTexture",X._matricesTexture,k),pt.setOptional(L,X,"batchingIdTexture"),pt.setValue(L,"batchingIdTexture",X._indirectTexture,k),pt.setOptional(L,X,"batchingColorTexture"),X._colorsTexture!==null&&pt.setValue(L,"batchingColorTexture",X._colorsTexture,k));const Oi=Y.morphAttributes;if((Oi.position!==void 0||Oi.normal!==void 0||Oi.color!==void 0)&&ue.update(X,Y,On),(yr||Fe.receiveShadow!==X.receiveShadow)&&(Fe.receiveShadow=X.receiveShadow,pt.setValue(L,"receiveShadow",X.receiveShadow)),(K.isMeshStandardMaterial||K.isMeshLambertMaterial||K.isMeshPhongMaterial)&&K.envMap===null&&O.environment!==null&&(zt.envMapIntensity.value=O.environmentIntensity),zt.dfgLUT!==void 0&&(zt.dfgLUT.value=NT()),yr&&(pt.setValue(L,"toneMappingExposure",C.toneMappingExposure),Fe.needsLights&&Ox(zt,Kr),Se&&K.fog===!0&&Ue.refreshFogUniforms(zt,Se),Ue.refreshMaterialUniforms(zt,K,Ae,ae,w.state.transmissionRenderTarget[T.id]),Pl.upload(L,ah(Fe),zt,k)),K.isShaderMaterial&&K.uniformsNeedUpdate===!0&&(Pl.upload(L,ah(Fe),zt,k),K.uniformsNeedUpdate=!1),K.isSpriteMaterial&&pt.setValue(L,"center",X.center),pt.setValue(L,"modelViewMatrix",X.modelViewMatrix),pt.setValue(L,"normalMatrix",X.normalMatrix),pt.setValue(L,"modelMatrix",X.matrixWorld),K.isShaderMaterial||K.isRawShaderMaterial){const Nt=K.uniformsGroups;for(let zi=0,Yr=Nt.length;zi<Yr;zi++){const ch=Nt[zi];Ee.update(ch,On),Ee.bind(ch,On)}}return On}function Ox(T,O){T.ambientLightColor.needsUpdate=O,T.lightProbe.needsUpdate=O,T.directionalLights.needsUpdate=O,T.directionalLightShadows.needsUpdate=O,T.pointLights.needsUpdate=O,T.pointLightShadows.needsUpdate=O,T.spotLights.needsUpdate=O,T.spotLightShadows.needsUpdate=O,T.rectAreaLights.needsUpdate=O,T.hemisphereLights.needsUpdate=O}function zx(T){return T.isMeshLambertMaterial||T.isMeshToonMaterial||T.isMeshPhongMaterial||T.isMeshStandardMaterial||T.isShadowMaterial||T.isShaderMaterial&&T.lights===!0}this.getActiveCubeFace=function(){return I},this.getActiveMipmapLevel=function(){return F},this.getRenderTarget=function(){return B},this.setRenderTargetTextures=function(T,O,Y){const K=S.get(T);K.__autoAllocateDepthBuffer=T.resolveDepthBuffer===!1,K.__autoAllocateDepthBuffer===!1&&(K.__useRenderToTexture=!1),S.get(T.texture).__webglTexture=O,S.get(T.depthTexture).__webglTexture=K.__autoAllocateDepthBuffer?void 0:Y,K.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(T,O){const Y=S.get(T);Y.__webglFramebuffer=O,Y.__useDefaultFramebuffer=O===void 0};const Bx=L.createFramebuffer();this.setRenderTarget=function(T,O=0,Y=0){B=T,I=O,F=Y;let K=null,X=!1,Se=!1;if(T){const Me=S.get(T);if(Me.__useDefaultFramebuffer!==void 0){Te.bindFramebuffer(L.FRAMEBUFFER,Me.__webglFramebuffer),G.copy(T.viewport),z.copy(T.scissor),j=T.scissorTest,Te.viewport(G),Te.scissor(z),Te.setScissorTest(j),W=-1;return}else if(Me.__webglFramebuffer===void 0)k.setupRenderTarget(T);else if(Me.__hasExternalTextures)k.rebindTextures(T,S.get(T.texture).__webglTexture,S.get(T.depthTexture).__webglTexture);else if(T.depthBuffer){const Xe=T.depthTexture;if(Me.__boundDepthTexture!==Xe){if(Xe!==null&&S.has(Xe)&&(T.width!==Xe.image.width||T.height!==Xe.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");k.setupDepthRenderbuffer(T)}}const Pe=T.texture;(Pe.isData3DTexture||Pe.isDataArrayTexture||Pe.isCompressedArrayTexture)&&(Se=!0);const Ne=S.get(T).__webglFramebuffer;T.isWebGLCubeRenderTarget?(Array.isArray(Ne[O])?K=Ne[O][Y]:K=Ne[O],X=!0):T.samples>0&&k.useMultisampledRTT(T)===!1?K=S.get(T).__webglMultisampledFramebuffer:Array.isArray(Ne)?K=Ne[Y]:K=Ne,G.copy(T.viewport),z.copy(T.scissor),j=T.scissorTest}else G.copy(D).multiplyScalar(Ae).floor(),z.copy(q).multiplyScalar(Ae).floor(),j=ne;if(Y!==0&&(K=Bx),Te.bindFramebuffer(L.FRAMEBUFFER,K)&&Te.drawBuffers(T,K),Te.viewport(G),Te.scissor(z),Te.setScissorTest(j),X){const Me=S.get(T.texture);L.framebufferTexture2D(L.FRAMEBUFFER,L.COLOR_ATTACHMENT0,L.TEXTURE_CUBE_MAP_POSITIVE_X+O,Me.__webglTexture,Y)}else if(Se){const Me=O;for(let Pe=0;Pe<T.textures.length;Pe++){const Ne=S.get(T.textures[Pe]);L.framebufferTextureLayer(L.FRAMEBUFFER,L.COLOR_ATTACHMENT0+Pe,Ne.__webglTexture,Y,Me)}}else if(T!==null&&Y!==0){const Me=S.get(T.texture);L.framebufferTexture2D(L.FRAMEBUFFER,L.COLOR_ATTACHMENT0,L.TEXTURE_2D,Me.__webglTexture,Y)}W=-1},this.readRenderTargetPixels=function(T,O,Y,K,X,Se,we,Me=0){if(!(T&&T.isWebGLRenderTarget)){it("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Pe=S.get(T).__webglFramebuffer;if(T.isWebGLCubeRenderTarget&&we!==void 0&&(Pe=Pe[we]),Pe){Te.bindFramebuffer(L.FRAMEBUFFER,Pe);try{const Ne=T.textures[Me],Xe=Ne.format,Ye=Ne.type;if(T.textures.length>1&&L.readBuffer(L.COLOR_ATTACHMENT0+Me),!$e.textureFormatReadable(Xe)){it("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!$e.textureTypeReadable(Ye)){it("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}O>=0&&O<=T.width-K&&Y>=0&&Y<=T.height-X&&L.readPixels(O,Y,K,X,xe.convert(Xe),xe.convert(Ye),Se)}finally{const Ne=B!==null?S.get(B).__webglFramebuffer:null;Te.bindFramebuffer(L.FRAMEBUFFER,Ne)}}},this.readRenderTargetPixelsAsync=async function(T,O,Y,K,X,Se,we,Me=0){if(!(T&&T.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Pe=S.get(T).__webglFramebuffer;if(T.isWebGLCubeRenderTarget&&we!==void 0&&(Pe=Pe[we]),Pe)if(O>=0&&O<=T.width-K&&Y>=0&&Y<=T.height-X){Te.bindFramebuffer(L.FRAMEBUFFER,Pe);const Ne=T.textures[Me],Xe=Ne.format,Ye=Ne.type;if(T.textures.length>1&&L.readBuffer(L.COLOR_ATTACHMENT0+Me),!$e.textureFormatReadable(Xe))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!$e.textureTypeReadable(Ye))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const ke=L.createBuffer();L.bindBuffer(L.PIXEL_PACK_BUFFER,ke),L.bufferData(L.PIXEL_PACK_BUFFER,Se.byteLength,L.STREAM_READ),L.readPixels(O,Y,K,X,xe.convert(Xe),xe.convert(Ye),0);const ut=B!==null?S.get(B).__webglFramebuffer:null;Te.bindFramebuffer(L.FRAMEBUFFER,ut);const Ct=L.fenceSync(L.SYNC_GPU_COMMANDS_COMPLETE,0);return L.flush(),await Yy(L,Ct,4),L.bindBuffer(L.PIXEL_PACK_BUFFER,ke),L.getBufferSubData(L.PIXEL_PACK_BUFFER,0,Se),L.deleteBuffer(ke),L.deleteSync(Ct),Se}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(T,O=null,Y=0){const K=Math.pow(2,-Y),X=Math.floor(T.image.width*K),Se=Math.floor(T.image.height*K),we=O!==null?O.x:0,Me=O!==null?O.y:0;k.setTexture2D(T,0),L.copyTexSubImage2D(L.TEXTURE_2D,Y,0,0,we,Me,X,Se),Te.unbindTexture()};const jx=L.createFramebuffer(),Vx=L.createFramebuffer();this.copyTextureToTexture=function(T,O,Y=null,K=null,X=0,Se=0){let we,Me,Pe,Ne,Xe,Ye,ke,ut,Ct;const Et=T.isCompressedTexture?T.mipmaps[Se]:T.image;if(Y!==null)we=Y.max.x-Y.min.x,Me=Y.max.y-Y.min.y,Pe=Y.isBox3?Y.max.z-Y.min.z:1,Ne=Y.min.x,Xe=Y.min.y,Ye=Y.isBox3?Y.min.z:0;else{const zt=Math.pow(2,-X);we=Math.floor(Et.width*zt),Me=Math.floor(Et.height*zt),T.isDataArrayTexture?Pe=Et.depth:T.isData3DTexture?Pe=Math.floor(Et.depth*zt):Pe=1,Ne=0,Xe=0,Ye=0}K!==null?(ke=K.x,ut=K.y,Ct=K.z):(ke=0,ut=0,Ct=0);const dt=xe.convert(O.format),qt=xe.convert(O.type);let Fe;O.isData3DTexture?(k.setTexture3D(O,0),Fe=L.TEXTURE_3D):O.isDataArrayTexture||O.isCompressedArrayTexture?(k.setTexture2DArray(O,0),Fe=L.TEXTURE_2D_ARRAY):(k.setTexture2D(O,0),Fe=L.TEXTURE_2D),L.pixelStorei(L.UNPACK_FLIP_Y_WEBGL,O.flipY),L.pixelStorei(L.UNPACK_PREMULTIPLY_ALPHA_WEBGL,O.premultiplyAlpha),L.pixelStorei(L.UNPACK_ALIGNMENT,O.unpackAlignment);const vn=L.getParameter(L.UNPACK_ROW_LENGTH),ot=L.getParameter(L.UNPACK_IMAGE_HEIGHT),On=L.getParameter(L.UNPACK_SKIP_PIXELS),Zn=L.getParameter(L.UNPACK_SKIP_ROWS),yr=L.getParameter(L.UNPACK_SKIP_IMAGES);L.pixelStorei(L.UNPACK_ROW_LENGTH,Et.width),L.pixelStorei(L.UNPACK_IMAGE_HEIGHT,Et.height),L.pixelStorei(L.UNPACK_SKIP_PIXELS,Ne),L.pixelStorei(L.UNPACK_SKIP_ROWS,Xe),L.pixelStorei(L.UNPACK_SKIP_IMAGES,Ye);const Kr=T.isDataArrayTexture||T.isData3DTexture,pt=O.isDataArrayTexture||O.isData3DTexture;if(T.isDepthTexture){const zt=S.get(T),Oi=S.get(O),Nt=S.get(zt.__renderTarget),zi=S.get(Oi.__renderTarget);Te.bindFramebuffer(L.READ_FRAMEBUFFER,Nt.__webglFramebuffer),Te.bindFramebuffer(L.DRAW_FRAMEBUFFER,zi.__webglFramebuffer);for(let Yr=0;Yr<Pe;Yr++)Kr&&(L.framebufferTextureLayer(L.READ_FRAMEBUFFER,L.COLOR_ATTACHMENT0,S.get(T).__webglTexture,X,Ye+Yr),L.framebufferTextureLayer(L.DRAW_FRAMEBUFFER,L.COLOR_ATTACHMENT0,S.get(O).__webglTexture,Se,Ct+Yr)),L.blitFramebuffer(Ne,Xe,we,Me,ke,ut,we,Me,L.DEPTH_BUFFER_BIT,L.NEAREST);Te.bindFramebuffer(L.READ_FRAMEBUFFER,null),Te.bindFramebuffer(L.DRAW_FRAMEBUFFER,null)}else if(X!==0||T.isRenderTargetTexture||S.has(T)){const zt=S.get(T),Oi=S.get(O);Te.bindFramebuffer(L.READ_FRAMEBUFFER,jx),Te.bindFramebuffer(L.DRAW_FRAMEBUFFER,Vx);for(let Nt=0;Nt<Pe;Nt++)Kr?L.framebufferTextureLayer(L.READ_FRAMEBUFFER,L.COLOR_ATTACHMENT0,zt.__webglTexture,X,Ye+Nt):L.framebufferTexture2D(L.READ_FRAMEBUFFER,L.COLOR_ATTACHMENT0,L.TEXTURE_2D,zt.__webglTexture,X),pt?L.framebufferTextureLayer(L.DRAW_FRAMEBUFFER,L.COLOR_ATTACHMENT0,Oi.__webglTexture,Se,Ct+Nt):L.framebufferTexture2D(L.DRAW_FRAMEBUFFER,L.COLOR_ATTACHMENT0,L.TEXTURE_2D,Oi.__webglTexture,Se),X!==0?L.blitFramebuffer(Ne,Xe,we,Me,ke,ut,we,Me,L.COLOR_BUFFER_BIT,L.NEAREST):pt?L.copyTexSubImage3D(Fe,Se,ke,ut,Ct+Nt,Ne,Xe,we,Me):L.copyTexSubImage2D(Fe,Se,ke,ut,Ne,Xe,we,Me);Te.bindFramebuffer(L.READ_FRAMEBUFFER,null),Te.bindFramebuffer(L.DRAW_FRAMEBUFFER,null)}else pt?T.isDataTexture||T.isData3DTexture?L.texSubImage3D(Fe,Se,ke,ut,Ct,we,Me,Pe,dt,qt,Et.data):O.isCompressedArrayTexture?L.compressedTexSubImage3D(Fe,Se,ke,ut,Ct,we,Me,Pe,dt,Et.data):L.texSubImage3D(Fe,Se,ke,ut,Ct,we,Me,Pe,dt,qt,Et):T.isDataTexture?L.texSubImage2D(L.TEXTURE_2D,Se,ke,ut,we,Me,dt,qt,Et.data):T.isCompressedTexture?L.compressedTexSubImage2D(L.TEXTURE_2D,Se,ke,ut,Et.width,Et.height,dt,Et.data):L.texSubImage2D(L.TEXTURE_2D,Se,ke,ut,we,Me,dt,qt,Et);L.pixelStorei(L.UNPACK_ROW_LENGTH,vn),L.pixelStorei(L.UNPACK_IMAGE_HEIGHT,ot),L.pixelStorei(L.UNPACK_SKIP_PIXELS,On),L.pixelStorei(L.UNPACK_SKIP_ROWS,Zn),L.pixelStorei(L.UNPACK_SKIP_IMAGES,yr),Se===0&&O.generateMipmaps&&L.generateMipmap(Fe),Te.unbindTexture()},this.initRenderTarget=function(T){S.get(T).__webglFramebuffer===void 0&&k.setupRenderTarget(T)},this.initTexture=function(T){T.isCubeTexture?k.setTextureCube(T,0):T.isData3DTexture?k.setTexture3D(T,0):T.isDataArrayTexture||T.isCompressedArrayTexture?k.setTexture2DArray(T,0):k.setTexture2D(T,0),Te.unbindTexture()},this.resetState=function(){I=0,F=0,B=null,Te.reset(),pe.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return ai}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const n=this.getContext();n.drawingBufferColorSpace=rt._getDrawingBufferColorSpace(e),n.unpackColorSpace=rt._getUnpackColorSpace()}}function kT(t){return t>3e4?"#9bb0ff":t>1e4?"#aabfff":t>7500?"#cad7ff":t>6e3?"#f8f7ff":t>5200?"#fffbe8":t>3700?"#ffd2a1":"#ffad51"}function OT(t,e){const n=(e||"").toLowerCase();return t>11||n.includes("jupiter")?"#c9956c":t>6||n.includes("saturn")?"#e8d9a0":t>3||n.includes("neptune")?"#5b9bd5":t>1.6||n.includes("super")?"#7fba6e":"#4d88bb"}function zT(t,e){const n=(e||"").toLowerCase();return t>6||n.includes("saturn")||n.includes("jupiter")||n.includes("gazeuse")}function BT({data:t,nasaPlanets:e}){const n=Z.useRef(null),i=!!t,r=(t==null?void 0:t.score)??.5,s=(t==null?void 0:t.characterization)||{},o=(t==null?void 0:t.metadata)||{},a=r>.7?"#4ade80":r>.4?"#fbbf24":"#f87171",c=e!=null&&e.length?e.filter(u=>u.period_days).map(u=>({name:u.name||"Planet",period_days:u.period_days,radius_earth:u.radius_earth||2,planet_type:null})):[{name:(t==null?void 0:t.target)||"Planet",period_days:(t==null?void 0:t.period_days)||10,radius_earth:s.planet_radius_earth||2,planet_type:s.planet_type||null}];return Z.useEffect(()=>{const u=n.current;if(!u)return;const p=new fS,h=u.clientWidth,f=u.clientHeight,g=new Mn(45,h/f,.1,300);g.position.set(0,6,15),g.lookAt(0,0,0);const x=new UT({antialias:!0,alpha:!1});x.setSize(h,f),x.setPixelRatio(Math.min(window.devicePixelRatio,2)),x.setClearColor(132104),u.appendChild(x.domElement);const M=[];for(let ve=0;ve<2400;ve++){const Ve=60+Math.random()*90,ie=Math.random()*Math.PI*2,le=Math.acos(2*Math.random()-1);M.push(Ve*Math.sin(le)*Math.cos(ie),Ve*Math.sin(le)*Math.sin(ie),Ve*Math.cos(le))}const v=new rn;v.setAttribute("position",new Wt(M,3)),p.add(new TS(v,new cx({color:16777215,size:.11,sizeAttenuation:!0}))),p.add(new zS(1122884,.6));const d=o.star_temperature_k||5778,m=o.star_radius_solar||1,_=Math.max(.8,Math.min(2.5,m*1.1)),b=kT(d),w=new tt(b),A=new pn(new Rs(_,64,64),new As({color:w}));p.add(A);const E=new pn(new Rs(_*1.06,32,32),new As({color:w,transparent:!0,opacity:.18,side:en}));p.add(E);const y=document.createElement("canvas");y.width=256,y.height=256;const C=y.getContext("2d"),P=parseInt(b.slice(1,3),16),I=parseInt(b.slice(3,5),16),F=parseInt(b.slice(5,7),16),B=C.createRadialGradient(128,128,0,128,128,128);B.addColorStop(0,`rgba(${P},${I},${F},0.85)`),B.addColorStop(.2,`rgba(${P},${I},${F},0.4)`),B.addColorStop(.5,`rgba(${P},${I},${F},0.1)`),B.addColorStop(1,`rgba(${P},${I},${F},0)`),C.fillStyle=B,C.fillRect(0,0,256,256);const W=new vS(new ox({map:new wS(y),transparent:!0,blending:Bd}));W.scale.set(_*8,_*8,1),p.add(W),p.add(new OS(w,4.5,90,1.2));const V=Math.max(...c.map(ve=>ve.period_days)),G=_+1.8,j=((c.length===1?_+5:10)-G)/Math.pow(V,2/3),$=c.map((ve,Ve)=>{const ie=ve.radius_earth||2,le=Math.max(.1,Math.min(.75,ie*.063)),ze=G+Math.pow(ve.period_days,2/3)*j,L=Math.max(4,Math.min(30,ve.period_days*1.3)),Re=Ve/c.length*Math.PI*2,Ge=new tt(OT(ie,ve.planet_type)),$e=new pn(new Rs(le,48,48),new LS({color:Ge,roughness:.6,metalness:.08}));if($e.add(new pn(new Rs(le*1.18,32,32),new As({color:Ge,transparent:!0,opacity:.07,side:en}))),zT(ie,ve.planet_type)){const R=new qp(le*1.5,le*2.5,80),S=R.attributes.position,k=R.attributes.uv,te=le*1.5,re=le*2.5;for(let be=0;be<S.count;be++){const ge=Math.sqrt(S.getX(be)**2+S.getY(be)**2);k.setXY(be,(ge-te)/(re-te),0)}const J=new pn(R,new As({color:ie>8?13939843:12110040,side:ri,transparent:!0,opacity:.6}));J.rotation.x=Math.PI/2.4,$e.add(J)}p.add($e);const Te=[];for(let R=0;R<=128;R++){const S=R/128*Math.PI*2;Te.push(new H(Math.cos(S)*ze,0,Math.sin(S)*ze))}return p.add(new ES(new rn().setFromPoints(Te),new lx({color:3364266,transparent:!0,opacity:.18}))),{mesh:$e,orbitR:ze,periodSec:L,phase:Re}});let Q=!1,se={x:0,y:0},ae=0,Ae=.4,De=15;const Oe=ve=>{Q=!0,se={x:ve.clientX,y:ve.clientY}},D=()=>{Q=!1},q=ve=>{Q&&(ae-=(ve.clientX-se.x)*.005,Ae=Math.max(-1.2,Math.min(1.2,Ae+(ve.clientY-se.y)*.005)),se={x:ve.clientX,y:ve.clientY})},ne=ve=>{ve.preventDefault(),De=Math.max(4,Math.min(32,De+ve.deltaY*.012))};x.domElement.addEventListener("pointerdown",Oe),x.domElement.addEventListener("pointerup",D),x.domElement.addEventListener("pointermove",q),x.domElement.addEventListener("wheel",ne,{passive:!1});let oe;const ye=new VS,Le=()=>{oe=requestAnimationFrame(Le);const ve=ye.getElapsedTime();$.forEach(ie=>{const le=ve/ie.periodSec*Math.PI*2+ie.phase;ie.mesh.position.set(Math.cos(le)*ie.orbitR,0,Math.sin(le)*ie.orbitR),ie.mesh.rotation.y=ve*.4}),A.rotation.y=ve*.05,E.rotation.y=ve*.04;const Ve=ve*.07+ae;g.position.set(Math.sin(Ve)*Math.cos(Ae)*De,Math.sin(Ae)*De,Math.cos(Ve)*Math.cos(Ae)*De),g.lookAt(0,0,0),x.render(p,g)};Le();const ht=()=>{const ve=u.clientWidth,Ve=u.clientHeight;g.aspect=ve/Ve,g.updateProjectionMatrix(),x.setSize(ve,Ve)};return window.addEventListener("resize",ht),()=>{cancelAnimationFrame(oe),window.removeEventListener("resize",ht),x.domElement.removeEventListener("pointerdown",Oe),x.domElement.removeEventListener("pointerup",D),x.domElement.removeEventListener("pointermove",q),x.domElement.removeEventListener("wheel",ne),x.dispose(),u.contains(x.domElement)&&u.removeChild(x.domElement)}},[t,e]),l.jsxs("div",{style:{position:"relative",width:"100%",height:"100%",minHeight:340},children:[l.jsx("div",{ref:n,style:{position:"absolute",inset:0,borderRadius:14,overflow:"hidden",background:"#020408"}}),l.jsxs("div",{style:{position:"absolute",top:12,left:14,zIndex:5,pointerEvents:"none"},children:[l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace",marginBottom:3},children:"Aperçu orbital 3D"}),l.jsx("div",{style:{fontSize:14,fontWeight:600,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:i?t.target:"En attente..."}),(e==null?void 0:e.length)>1&&l.jsxs("div",{style:{fontSize:10,color:"rgba(74,222,160,0.5)",fontFamily:"'DM Mono',monospace",marginTop:3},children:[e.length," planètes · NASA confirmées"]})]}),i&&l.jsxs("div",{style:{position:"absolute",top:12,right:14,zIndex:5,padding:"4px 10px",borderRadius:999,fontSize:10,color:a,background:`${a}16`,border:`1px solid ${a}33`,fontFamily:"'DM Mono',monospace",backdropFilter:"blur(4px)"},children:[(r*100).toFixed(1),"% confiance IA"]}),i&&l.jsx("div",{style:{position:"absolute",bottom:28,left:14,right:14,zIndex:5,display:"flex",gap:8,flexWrap:"wrap",pointerEvents:"none"},children:[s.planet_type&&{label:"Type",value:s.planet_type},s.planet_radius_earth&&{label:"Rayon",value:`${s.planet_radius_earth} R⊕`},t.period_days&&{label:"Période",value:`${t.period_days} j`},o.star_temperature_k&&{label:"Étoile",value:`${o.star_temperature_k.toLocaleString()} K`}].filter(Boolean).map((u,p)=>l.jsxs("div",{style:{padding:"3px 9px",borderRadius:6,background:"rgba(4,7,18,0.82)",backdropFilter:"blur(4px)",border:"1px solid rgba(91,141,239,0.12)",fontFamily:"'DM Mono',monospace",fontSize:9},children:[l.jsxs("span",{style:{color:"rgba(160,180,220,0.4)"},children:[u.label," "]}),l.jsx("span",{style:{color:"#e4e8f7"},children:u.value})]},p))}),l.jsx("div",{style:{position:"absolute",bottom:8,right:14,zIndex:5,fontSize:9,color:"rgba(160,180,220,0.22)",fontFamily:"'DM Mono',monospace"},children:"Glisser pour tourner · Molette pour zoomer"})]})}/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const jT=t=>t.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),VT=t=>t.replace(/^([A-Z])|[\s-_]+(\w)/g,(e,n,i)=>i?i.toUpperCase():n.toLowerCase()),_0=t=>{const e=VT(t);return e.charAt(0).toUpperCase()+e.slice(1)},Mx=(...t)=>t.filter((e,n,i)=>!!e&&e.trim()!==""&&i.indexOf(e)===n).join(" ").trim(),HT=t=>{for(const e in t)if(e.startsWith("aria-")||e==="role"||e==="title")return!0};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var GT={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const WT=Z.forwardRef(({color:t="currentColor",size:e=24,strokeWidth:n=2,absoluteStrokeWidth:i,className:r="",children:s,iconNode:o,...a},c)=>Z.createElement("svg",{ref:c,...GT,width:e,height:e,stroke:t,strokeWidth:i?Number(n)*24/Number(e):n,className:Mx("lucide",r),...!s&&!HT(a)&&{"aria-hidden":"true"},...a},[...o.map(([u,p])=>Z.createElement(u,p)),...Array.isArray(s)?s:[s]]));/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Je=(t,e)=>{const n=Z.forwardRef(({className:i,...r},s)=>Z.createElement(WT,{ref:s,iconNode:e,className:Mx(`lucide-${jT(_0(t))}`,`lucide-${t}`,i),...r}));return n.displayName=_0(t),n};/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const XT=[["path",{d:"M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2",key:"169zse"}]],$p=Je("activity",XT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const qT=[["path",{d:"M12 7v14",key:"1akyts"}],["path",{d:"M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z",key:"ruj8y"}]],Kp=Je("book-open",qT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $T=[["path",{d:"M5 21v-6",key:"1hz6c0"}],["path",{d:"M12 21V3",key:"1lcnhd"}],["path",{d:"M19 21V9",key:"unv183"}]],bx=Je("chart-no-axes-column",$T);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const KT=[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]],ra=Je("chevron-right",KT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const YT=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]],ma=Je("circle-check",YT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ZT=[["path",{d:"M12 6v6l4 2",key:"mmk7yg"}],["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}]],Uf=Je("clock",ZT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const JT=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M12 3v18",key:"108xh3"}]],Ex=Je("columns-2",JT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const QT=[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 21 19V5",key:"1wlel7"}],["path",{d:"M3 12A9 3 0 0 0 21 12",key:"mv7ke4"}]],Yp=Je("database",QT);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const e3=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",ry:"2",key:"1m3agn"}],["path",{d:"M16 8h.01",key:"cr5u4v"}],["path",{d:"M16 12h.01",key:"1l6xoz"}],["path",{d:"M16 16h.01",key:"1f9h7w"}],["path",{d:"M8 8h.01",key:"1e4136"}],["path",{d:"M8 12h.01",key:"czm47f"}],["path",{d:"M8 16h.01",key:"18s6g9"}]],t3=Je("dice-6",e3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const n3=[["path",{d:"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49",key:"ct8e1f"}],["path",{d:"M14.084 14.158a3 3 0 0 1-4.242-4.242",key:"151rxh"}],["path",{d:"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143",key:"13bj9a"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]],kf=Je("eye-off",n3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const i3=[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],Of=Je("eye",i3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const r3=[["path",{d:"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z",key:"1oefj6"}],["path",{d:"M14 2v5a1 1 0 0 0 1 1h5",key:"wfsgrz"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]],Tx=Je("file-text",r3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const s3=[["path",{d:"M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z",key:"sc7q7i"}]],o3=Je("funnel",s3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const a3=[["path",{d:"M9 10h.01",key:"qbtxuw"}],["path",{d:"M15 10h.01",key:"1qmjsl"}],["path",{d:"M12 2a8 8 0 0 0-8 8v12l3-3 2.5 2.5L12 19l2.5 2.5L17 19l3 3V10a8 8 0 0 0-8-8z",key:"uwwb07"}]],l3=Je("ghost",a3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const c3=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20",key:"13o1zl"}],["path",{d:"M2 12h20",key:"9i4pu4"}]],Zp=Je("globe",c3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const u3=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]],Lc=Je("info",u3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const d3=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],mi=Je("loader-circle",d3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const f3=[["rect",{width:"18",height:"11",x:"3",y:"11",rx:"2",ry:"2",key:"1w4ew1"}],["path",{d:"M7 11V7a5 5 0 0 1 10 0v4",key:"fwvmzm"}]],wx=Je("lock",f3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const p3=[["path",{d:"m10 17 5-5-5-5",key:"1bsop3"}],["path",{d:"M15 12H3",key:"6jk70r"}],["path",{d:"M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4",key:"u53s6r"}]],y0=Je("log-in",p3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const h3=[["path",{d:"m16 17 5-5-5-5",key:"1bji2h"}],["path",{d:"M21 12H9",key:"dn1m92"}],["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}]],m3=Je("log-out",h3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const g3=[["rect",{width:"20",height:"14",x:"2",y:"3",rx:"2",key:"48i651"}],["line",{x1:"8",x2:"16",y1:"21",y2:"21",key:"1svkeh"}],["line",{x1:"12",x2:"12",y1:"17",y2:"21",key:"vw1qmm"}]],x3=Je("monitor",g3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const v3=[["path",{d:"M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401",key:"kfwtm"}]],_3=Je("moon",v3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const y3=[["path",{d:"M20.341 6.484A10 10 0 0 1 10.266 21.85",key:"1enhxb"}],["path",{d:"M3.659 17.516A10 10 0 0 1 13.74 2.152",key:"1crzgf"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}],["circle",{cx:"19",cy:"5",r:"2",key:"mhkx31"}],["circle",{cx:"5",cy:"19",r:"2",key:"v8kfzx"}]],ga=Je("orbit",y3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const S3=[["path",{d:"M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z",key:"m3kijz"}],["path",{d:"m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z",key:"1fmvmk"}],["path",{d:"M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0",key:"1f8sc4"}],["path",{d:"M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5",key:"qeys4"}]],M3=Je("rocket",S3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const b3=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]],Cx=Je("rotate-ccw",b3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const E3=[["path",{d:"m13.5 6.5-3.148-3.148a1.205 1.205 0 0 0-1.704 0L6.352 5.648a1.205 1.205 0 0 0 0 1.704L9.5 10.5",key:"dzhfyz"}],["path",{d:"M16.5 7.5 19 5",key:"1ltcjm"}],["path",{d:"m17.5 10.5 3.148 3.148a1.205 1.205 0 0 1 0 1.704l-2.296 2.296a1.205 1.205 0 0 1-1.704 0L13.5 14.5",key:"nfoymv"}],["path",{d:"M9 21a6 6 0 0 0-6-6",key:"1iajcf"}],["path",{d:"M9.352 10.648a1.205 1.205 0 0 0 0 1.704l2.296 2.296a1.205 1.205 0 0 0 1.704 0l4.296-4.296a1.205 1.205 0 0 0 0-1.704l-2.296-2.296a1.205 1.205 0 0 0-1.704 0z",key:"nv9zqy"}]],T3=Je("satellite",E3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const w3=[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]],Br=Je("search",w3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const C3=[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]],A3=Je("shield-check",C3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const R3=[["path",{d:"M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z",key:"1s2grr"}],["path",{d:"M20 2v4",key:"1rf3ol"}],["path",{d:"M22 4h-4",key:"gwowj6"}],["circle",{cx:"4",cy:"20",r:"2",key:"6kqj1y"}]],Fc=Je("sparkles",R3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const I3=[["path",{d:"M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z",key:"r04s7s"}]],sa=Je("star",I3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const P3=[["path",{d:"m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44",key:"k4qptu"}],["path",{d:"m13.56 11.747 4.332-.924",key:"19l80z"}],["path",{d:"m16 21-3.105-6.21",key:"7oh9d"}],["path",{d:"M16.485 5.94a2 2 0 0 1 1.455-2.425l1.09-.272a1 1 0 0 1 1.212.727l1.515 6.06a1 1 0 0 1-.727 1.213l-1.09.272a2 2 0 0 1-2.425-1.455z",key:"m7xp4m"}],["path",{d:"m6.158 8.633 1.114 4.456",key:"74o979"}],["path",{d:"m8 21 3.105-6.21",key:"1fvxut"}],["circle",{cx:"12",cy:"13",r:"2",key:"1c1ljs"}]],ur=Je("telescope",P3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const D3=[["path",{d:"M10 11v6",key:"nco0om"}],["path",{d:"M14 11v6",key:"outv1u"}],["path",{d:"M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6",key:"miytrc"}],["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",key:"e791ji"}]],L3=Je("trash-2",D3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const F3=[["path",{d:"M16 7h6v6",key:"box55l"}],["path",{d:"m22 7-8.5 8.5-5-5L2 17",key:"1t1m79"}]],Ax=Je("trending-up",F3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const N3=[["path",{d:"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3",key:"wmoenq"}],["path",{d:"M12 9v4",key:"juzpu7"}],["path",{d:"M12 17h.01",key:"p32p05"}]],hr=Je("triangle-alert",N3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const U3=[["path",{d:"M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2",key:"1yyitq"}],["circle",{cx:"9",cy:"7",r:"4",key:"nufk8"}],["line",{x1:"19",x2:"19",y1:"8",y2:"14",key:"1bvyxn"}],["line",{x1:"22",x2:"16",y1:"11",y2:"11",key:"1shjgl"}]],S0=Je("user-plus",U3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const k3=[["path",{d:"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2",key:"975kel"}],["circle",{cx:"12",cy:"7",r:"4",key:"17ys0d"}]],Ys=Je("user",k3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const O3=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],Rx=Je("x",O3);/**
 * @license lucide-react v0.562.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const z3=[["path",{d:"M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z",key:"1xq2db"}]],Nc=Je("zap",z3),jt="http://localhost:5001",xa=Z.createContext(!1),B3=[{id:"Kepler-10",label:"Kepler-10"},{id:"Kepler-22",label:"Kepler-22"},{id:"Kepler-90",label:"Kepler-90"},{id:"Kepler-452",label:"Kepler-452"},{id:"Kepler-62",label:"Kepler-62"},{id:"Kepler-186",label:"Kepler-186"}],j3=["TIC 231670397","TIC 16288184","TIC 144065872","TIC 66818296","TIC 317060587","TIC 366989877","TIC 320004517","TIC 38846515","TIC 149601557","TIC 400595342","TIC 304021498","TIC 363260203","TIC 261136679","TIC 55525572","TIC 192826603"],Uc=["KIC 10000490","KIC 10023469","KIC 10091257","KIC 10154388","KIC 10203349","KIC 10268714","KIC 10330115","KIC 10384798","KIC 10460984","KIC 10514429","KIC 10577994","KIC 10657406","KIC 10709622","KIC 10753922","KIC 10874614","KIC 10963065","KIC 11027624","KIC 11080405","KIC 11187436","KIC 11236244","KIC 11304987","KIC 11403530","KIC 11463211","KIC 11521793","KIC 11621897","KIC 11709124","KIC 11818872","KIC 11918099","KIC 12010534","KIC 12216278","KIC 12555140","KIC 2010191","KIC 2444412","KIC 2574201","KIC 2849805","KIC 3114167","KIC 3239945","KIC 3342467","KIC 3448130","KIC 3644399","KIC 3742855","KIC 3851193","KIC 3965326","KIC 4076976","KIC 4164994","KIC 4262581","KIC 4385148","KIC 4545187","KIC 4664743","KIC 4757437","KIC 4843751","KIC 4917596","KIC 5036480","KIC 5094751","KIC 5181455","KIC 5286786","KIC 5385410","KIC 5471202","KIC 5513897","KIC 5551504","KIC 5652237","KIC 5738346","KIC 5818068","KIC 5955621","KIC 6034945","KIC 6062929","KIC 6185331","KIC 6263593","KIC 6311520","KIC 6364582","KIC 6437617","KIC 6528464","KIC 6600492","KIC 6665064","KIC 6705026","KIC 6776401","KIC 6929841","KIC 7024045","KIC 7047922","KIC 7115597","KIC 7185710","KIC 7283710","KIC 7379385","KIC 7463685","KIC 7542369","KIC 7663405","KIC 7743464","KIC 7838675","KIC 7907423","KIC 8012732","KIC 8043638","KIC 8106610","KIC 8155368","KIC 8222813","KIC 8246781","KIC 8278371","KIC 8358012","KIC 8414914","KIC 8487645","KIC 8552719","KIC 8608544","KIC 8644288","KIC 8733898","KIC 8766222","KIC 8826878","KIC 8890150","KIC 8953257","KIC 9034103","KIC 9117416","KIC 9166870","KIC 9291039","KIC 9351920","KIC 9412445","KIC 9474483","KIC 9529733","KIC 9593528","KIC 9652649","KIC 9714550","KIC 9777090","KIC 9824805"],Jp=["Kepler-10","Kepler-10b","Kepler-10c","Kepler-11","Kepler-16","Kepler-16b","Kepler-20","Kepler-20f","Kepler-22","Kepler-22b","Kepler-25","Kepler-36","Kepler-37","Kepler-42","Kepler-47","Kepler-55","Kepler-62","Kepler-62f","Kepler-68","Kepler-69","Kepler-78","Kepler-80","Kepler-89","Kepler-90","Kepler-93","Kepler-102","Kepler-138","Kepler-160","Kepler-167","Kepler-186","Kepler-186f","Kepler-296","Kepler-395","Kepler-421","Kepler-438","Kepler-438b","Kepler-442","Kepler-442b","Kepler-444","Kepler-452","Kepler-452b","Kepler-453","Kepler-503","Kepler-560"],Dl=[{sel:null,title:"🚀 Bienvenue !",desc:"Ce tutoriel rapide te présente toutes les fonctionnalités en moins de 2 minutes. Clique sur Suivant pour commencer, ou Passer pour ignorer."},{sel:"[data-tour='mode-toggle']",title:"✨ Débutant / Expert",desc:"Choisis ton niveau ici. Le mode Débutant simplifie tout en français courant avec des emojis. Le mode Expert affiche les données scientifiques complètes."},{sel:"[data-tour='nav']",title:"🗂 Les onglets",desc:"Chaque onglet est un outil différent. Tu peux naviguer librement entre Analyse, Comparaison, Catalogue, Historique et Documentation."},{sel:"[data-tour='search']",title:"🔍 Analyser une étoile",desc:"Tape le nom d'une étoile (ex: Kepler-22b) ou son identifiant KIC. L'IA analyse sa courbe de lumière et te dit si une planète est probable — en quelques secondes."},{sel:"[data-tour='tab-comparison']",title:"⚖️ Comparaison",desc:"Compare jusqu'à 3 étoiles côte à côte : courbes de lumière, score IA et caractéristiques orbitales."},{sel:"[data-tour='tab-catalog']",title:"📚 Catalogue",desc:"Parcours toutes nos étoiles avec des filtres avancés (SNR, période, type planète), ou upload ton propre fichier CSV pour analyser une étoile personnalisée."},{sel:"[data-tour='tab-history']",title:"🕓 Historique",desc:"Retrouve toutes tes analyses passées, même après déconnexion. Tu peux relancer une analyse directement depuis ici."},{sel:null,title:"🎉 C'est parti !",desc:"Tu connais maintenant toutes les fonctionnalités. Lance-toi en tapant un nom d'étoile dans la barre de recherche !"}],zf=[{key:"connect",label:"Connexion API",pct:10},{key:"acquire",label:"Téléchargement courbe",pct:30},{key:"preprocess",label:"Prétraitement signal",pct:55},{key:"features",label:"Extraction features",pct:75},{key:"predict",label:"Prédiction XGBoost",pct:90},{key:"done",label:"Terminé",pct:100}];let Qp=null;const Ix=()=>Qp,V3=t=>{Qp=t},Ll=()=>{Qp=null};async function ln(t,e={}){const n=Ix();if(!n)throw new Error("Non authentifié");const i={...e.headers,Authorization:`Bearer ${n.token}`};e.body&&typeof e.body=="string"&&!i["Content-Type"]&&(i["Content-Type"]="application/json");const r=await fetch(t,{...e,headers:i});if(r.status===401)throw Ll(),new Error("Session expirée");return r}const Px=`
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-thumb { background: rgba(91,141,239,.15); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(91,141,239,.3); }
  @keyframes twinkle  { 0%{opacity:.08} 100%{opacity:.6} }
  @keyframes spin     { 100%{transform:rotate(360deg)} }
  @keyframes fadeIn   { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideIn  { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
  @keyframes pulse    { 0%,100%{box-shadow:0 0 18px rgba(91,141,239,.06)} 50%{box-shadow:0 0 36px rgba(91,141,239,.15)} }
  @keyframes breathe  { 0%,100%{opacity:.4} 50%{opacity:1} }
  @keyframes stellar-glow {
    0%,100%{box-shadow:0 0 20px rgba(240,192,64,.2),0 0 40px rgba(240,192,64,.08)}
    50%{box-shadow:0 0 30px rgba(240,192,64,.35),0 0 60px rgba(240,192,64,.12)}
  }
  @keyframes transit-orbit {
    0%  {transform:translate(calc(-50% + 60px),-50%);opacity:0}
    15% {opacity:1}
    50% {transform:translate(-50%,-50%);opacity:1}
    85% {opacity:1}
    100%{transform:translate(calc(-50% - 60px),-50%);opacity:0}
  }
  @keyframes mini-orbit {
    0%  {top:2px;left:50%;opacity:0.8}
    25% {top:50%;left:calc(100% - 2px);opacity:1}
    50% {top:calc(100% - 2px);left:50%;opacity:0.8}
    75% {top:50%;left:2px;opacity:1}
    100%{top:2px;left:50%;opacity:0.8}
  }
  @keyframes gradient-shift {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
  }
  @keyframes nebula-drift {
    0%,100%{opacity:.03;transform:scale(1)}
    50%{opacity:.06;transform:scale(1.1)}
  }
`;function Dx(){const t=Z.useRef(Array.from({length:110},()=>({x:Math.random()*100,y:Math.random()*100,s:.5+Math.random()*1.5,o:.15+Math.random()*.55,d:Math.random()*4}))).current;return l.jsx("div",{style:{position:"fixed",inset:0,pointerEvents:"none",zIndex:0,overflow:"hidden"},children:t.map((e,n)=>l.jsx("div",{style:{position:"absolute",left:`${e.x}%`,top:`${e.y}%`,width:e.s,height:e.s,borderRadius:"50%",background:"#fff",opacity:e.o,animation:`twinkle ${2+e.d}s ease-in-out infinite alternate`,animationDelay:`${e.d}s`}},n))})}function Qe({children:t,style:e={},glow:n=!1,onClick:i}){return l.jsx("div",{onClick:i,style:{background:"rgba(10,13,22,0.75)",backdropFilter:"blur(16px)",border:"1px solid rgba(91,141,239,0.1)",borderRadius:14,padding:16,animation:n?"pulse 6s ease-in-out infinite":void 0,...e},children:t})}function H3({progress:t}){if(!(t!=null&&t.visible))return null;const{stepIdx:e,pct:n,waiting:i}=t,r=n>=100,s=r?"#34d399":"#5b8def";return l.jsxs(Qe,{style:{animation:"fadeIn .4s ease-out"},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8},children:[r?l.jsx(ma,{size:14,style:{color:"#34d399"}}):l.jsx(mi,{size:14,style:{color:"#5b8def",animation:"spin 1s linear infinite"}}),l.jsx("span",{style:{fontSize:12,fontWeight:600,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:"Pipeline d'analyse"})]}),l.jsxs("span",{style:{fontSize:18,fontWeight:700,fontFamily:"'DM Mono',monospace",color:s},children:[n,"%"]})]}),l.jsx("div",{style:{height:4,borderRadius:2,background:"rgba(91,141,239,0.1)",marginBottom:12,overflow:"hidden"},children:l.jsx("div",{style:{height:"100%",width:`${n}%`,borderRadius:2,background:`linear-gradient(90deg,${s},#7c3aed)`,transition:"width 0.5s cubic-bezier(0.22,1,0.36,1)",boxShadow:`0 0 10px ${s}40`}})}),l.jsx("div",{style:{display:"flex",gap:5,flexWrap:"wrap"},children:zf.map((o,a)=>{const c=a<e,u=a===e,p=c?"#34d399":u?"#5b8def":"rgba(160,180,220,0.2)";return l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:4,padding:"3px 9px",borderRadius:6,fontSize:10,fontFamily:"'DM Mono',monospace",color:p,background:c?"rgba(52,211,153,0.08)":u?"rgba(91,141,239,0.1)":"rgba(15,18,30,0.5)",border:`1px solid ${p}25`},children:[l.jsx("span",{children:c?"✓":`0${a+1}`})," ",o.label]},o.key)})}),i&&!r&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,marginTop:10,padding:"6px 12px",borderRadius:8,background:"rgba(91,141,239,0.05)",border:"1px solid rgba(91,141,239,0.1)"},children:[l.jsx("div",{style:{width:6,height:6,borderRadius:"50%",background:"#5b8def",animation:"breathe 1.5s ease-in-out infinite",flexShrink:0}}),l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",animation:"breathe 1.5s ease-in-out infinite"},children:"En attente du résultat…"})]})]})}function dc({data:t,score:e,isLoading:n}){const i=Z.useRef(null),[r,s]=Z.useState(null),o=Z.useRef(0),[a,c]=Z.useState(null),u=Z.useRef(null),p=a!==null,h=Z.useMemo(()=>{if(!t||t.length===0)return null;const w=t.map(F=>F.time),A=t.map(F=>F.flux),E=Math.min(...w),y=Math.max(...w),C=Math.min(...A),P=Math.max(...A),I=(P-C)*.1||.001;return{tMin:E,tMax:y,fMin:C-I,fMax:P+I}},[t]);Z.useEffect(()=>{c(null)},[t]);const f=Z.useCallback((w=1)=>{const A=i.current;if(!A||!t||t.length===0||!h)return;const E=A.getContext("2d"),y=window.devicePixelRatio||1,C=A.getBoundingClientRect();A.width=C.width*y,A.height=C.height*y,E.scale(y,y);const P=C.width,I=C.height,F={top:30,right:24,bottom:46,left:68},B=P-F.left-F.right,W=I-F.top-F.bottom;E.fillStyle="#07090f",E.fillRect(0,0,P,I);const V=a||h,{tMin:G,tMax:z,fMin:j,fMax:$}=V,Q=z-G||1,se=$-j||.001,ae=oe=>F.left+(oe-G)/Q*B,Ae=oe=>F.top+W-(oe-j)/se*W;E.strokeStyle="rgba(91,141,239,0.05)",E.lineWidth=1;for(let oe=0;oe<=5;oe++){const ye=F.top+W/5*oe;E.beginPath(),E.moveTo(F.left,ye),E.lineTo(P-F.right,ye),E.stroke()}for(let oe=0;oe<=6;oe++){const ye=F.left+B/6*oe;E.beginPath(),E.moveTo(ye,F.top),E.lineTo(ye,I-F.bottom),E.stroke()}E.fillStyle="rgba(160,180,220,0.45)",E.font="10px 'DM Mono',monospace",E.textAlign="center";for(let oe=0;oe<=6;oe++)E.fillText((G+Q/6*oe).toFixed(3),F.left+B/6*oe,I-F.bottom+16);E.textAlign="right";for(let oe=0;oe<=5;oe++)E.fillText((j+se/5*(5-oe)).toFixed(5),F.left-6,F.top+W/5*oe+4);E.fillStyle="rgba(160,180,220,0.5)",E.font="11px 'DM Mono',monospace",E.textAlign="center",E.fillText("Phase Orbitale",P/2,I-4),E.save(),E.translate(12,I/2),E.rotate(-Math.PI/2),E.fillText("Flux Relatif",0,0),E.restore();const De=t.reduce((oe,ye)=>ye.flux<oe.flux?ye:oe,t[0]),Oe=ae(De.time);if(Oe>=F.left&&Oe<=P-F.right){const oe=E.createRadialGradient(Oe,Ae(De.flux),0,Oe,Ae(De.flux),90);oe.addColorStop(0,"rgba(91,141,239,0.07)"),oe.addColorStop(1,"rgba(91,141,239,0)"),E.fillStyle=oe,E.fillRect(F.left,F.top,B,W)}E.save(),E.beginPath(),E.rect(F.left,F.top,B,W),E.clip();const D=Math.floor(t.length*w),q=e>=.7?"rgba(52,211,153,0.65)":e>=.35?"rgba(251,191,36,0.65)":"rgba(248,113,113,0.65)",ne=e>=.7?"rgba(52,211,153,0.14)":e>=.35?"rgba(251,191,36,0.14)":"rgba(248,113,113,0.14)";for(let oe=0;oe<D;oe++){const ye=ae(t[oe].time),Le=Ae(t[oe].flux);E.beginPath(),E.arc(ye,Le,3.5,0,Math.PI*2),E.fillStyle=ne,E.fill(),E.beginPath(),E.arc(ye,Le,1.4,0,Math.PI*2),E.fillStyle=q,E.fill()}E.restore(),w>=1&&Oe>=F.left&&Oe<=P-F.right&&(E.setLineDash([3,4]),E.strokeStyle="rgba(91,141,239,0.35)",E.lineWidth=1,E.beginPath(),E.moveTo(Oe,F.top),E.lineTo(Oe,I-F.bottom),E.stroke(),E.setLineDash([]),E.fillStyle="rgba(91,141,239,0.9)",E.font="9px 'DM Mono',monospace",E.textAlign="center",E.fillText("▼ Transit",Oe,F.top-8))},[t,e,a,h]);Z.useEffect(()=>{if(!t||t.length===0)return;let w=null;const A=E=>{w||(w=E);const y=Math.min((E-w)/1100,1);f(1-(1-y)**3),y<1&&(o.current=requestAnimationFrame(A))};return o.current=requestAnimationFrame(A),()=>cancelAnimationFrame(o.current)},[t]),Z.useEffect(()=>{f(1)},[f]);const g=Z.useCallback((w,A)=>{const E=i.current;if(!E||!h)return null;const y=E.getBoundingClientRect(),C={top:30,right:24,bottom:46,left:68},P=y.width-C.left-C.right,I=y.height-C.top-C.bottom,F=a||h;return{t:F.tMin+(w-C.left)/P*(F.tMax-F.tMin),f:F.fMin+(I-(A-C.top))/I*(F.fMax-F.fMin)}},[a,h]),x=Z.useCallback(w=>{if(w.preventDefault(),!h)return;const E=i.current.getBoundingClientRect(),y=g(w.clientX-E.left,w.clientY-E.top);if(!y)return;const C=w.deltaY<0?.75:1/.75,P=a||h,I=y.t-(y.t-P.tMin)*C,F=y.t+(P.tMax-y.t)*C,B=y.f-(y.f-P.fMin)*C,W=y.f+(P.fMax-y.f)*C,V=h;if(F-I>=(V.tMax-V.tMin)*1.05&&W-B>=(V.fMax-V.fMin)*1.05){c(null);return}c({tMin:I,tMax:F,fMin:B,fMax:W})},[a,h,g]),M=Z.useCallback(w=>{!t||!t.length||w.button!==0||(u.current={startX:w.clientX,startY:w.clientY,vpSnap:a||h})},[a,h,t]),v=Z.useCallback(w=>{if(!t||!t.length)return;if(u.current){const P=i.current.getBoundingClientRect(),I={top:30,right:24,bottom:46,left:68},F=P.width-I.left-I.right,B=P.height-I.top-I.bottom,W=u.current.vpSnap,V=-((w.clientX-u.current.startX)/F)*(W.tMax-W.tMin),G=(w.clientY-u.current.startY)/B*(W.fMax-W.fMin);c({tMin:W.tMin+V,tMax:W.tMax+V,fMin:W.fMin+G,fMax:W.fMax+G});return}const A=i.current.getBoundingClientRect(),E=g(w.clientX-A.left,w.clientY-A.top);if(!E)return;const y=t.reduce((C,P)=>Math.abs(P.time-E.t)<Math.abs(C.time-E.t)?P:C);s({x:w.clientX-A.left,y:w.clientY-A.top,time:y.time,flux:y.flux})},[t,a,h,g]),d=Z.useCallback(()=>{u.current=null},[]),m=Z.useCallback(()=>{u.current=null,s(null)},[]);Z.useEffect(()=>{const w=i.current;if(w)return w.addEventListener("wheel",x,{passive:!1}),()=>w.removeEventListener("wheel",x)},[x]);const _=()=>c(null),b=Z.useMemo(()=>{if(!a||!h)return 1;const w=h.tMax-h.tMin,A=a.tMax-a.tMin;return Math.max(1,Math.round(w/A))},[a,h]);return l.jsxs("div",{style:{position:"relative",width:"100%",height:"100%"},children:[n&&l.jsxs("div",{style:{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(7,9,15,0.75)",zIndex:10,borderRadius:12,gap:10},children:[l.jsx(mi,{size:28,style:{color:"#5b8def",animation:"spin 1s linear infinite"}}),l.jsx("span",{style:{color:"#5b8def",fontFamily:"'DM Mono',monospace",fontSize:13},children:"Analyse en cours…"})]}),(!t||t.length===0)&&!n&&l.jsxs("div",{style:{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:12},children:[l.jsxs("svg",{width:"120",height:"80",viewBox:"0 0 120 80",style:{opacity:.5},children:[l.jsx("circle",{cx:"60",cy:"40",r:"12",fill:"url(#starGrad)",opacity:"0.8",children:l.jsx("animate",{attributeName:"r",values:"11;13;11",dur:"3s",repeatCount:"indefinite"})}),l.jsxs("circle",{cx:"60",cy:"40",r:"20",fill:"none",stroke:"rgba(240,192,64,0.15)",strokeWidth:"1",children:[l.jsx("animate",{attributeName:"r",values:"18;22;18",dur:"3s",repeatCount:"indefinite"}),l.jsx("animate",{attributeName:"opacity",values:"0.3;0.1;0.3",dur:"3s",repeatCount:"indefinite"})]}),l.jsx("ellipse",{cx:"60",cy:"40",rx:"45",ry:"12",fill:"none",stroke:"rgba(91,141,239,0.15)",strokeWidth:"0.5",strokeDasharray:"3,3"}),l.jsx("circle",{r:"4",fill:"#1e2d4a",stroke:"rgba(91,141,239,0.3)",strokeWidth:"0.5",children:l.jsx("animateMotion",{dur:"5s",repeatCount:"indefinite",path:"M 15,0 A 45,12 0 1 1 14.9,0.1"})}),l.jsx("defs",{children:l.jsxs("radialGradient",{id:"starGrad",cx:"0.4",cy:"0.4",children:[l.jsx("stop",{offset:"0%",stopColor:"#fff8e1"}),l.jsx("stop",{offset:"50%",stopColor:"#f0c040"}),l.jsx("stop",{offset:"100%",stopColor:"#e8a020"})]})})]}),l.jsxs("div",{style:{textAlign:"center"},children:[l.jsx("div",{style:{color:"rgba(160,180,220,0.4)",fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600,marginBottom:4},children:"Pointez votre telescope"}),l.jsx("div",{style:{color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",fontSize:11},children:"Entrez un identifiant stellaire pour detecter des exoplanetes"})]})]}),t&&t.length>0&&l.jsxs("div",{style:{position:"absolute",top:8,right:8,display:"flex",alignItems:"center",gap:6,zIndex:15,pointerEvents:"none"},children:[p&&l.jsxs(l.Fragment,{children:[l.jsxs("div",{style:{padding:"2px 8px",borderRadius:5,fontSize:10,fontFamily:"'DM Mono',monospace",color:"#5b8def",background:"rgba(91,141,239,0.12)",border:"1px solid rgba(91,141,239,0.25)",backdropFilter:"blur(6px)"},children:["×",b]}),l.jsxs("button",{onClick:_,style:{pointerEvents:"all",display:"flex",alignItems:"center",gap:4,padding:"3px 8px",borderRadius:5,border:"1px solid rgba(91,141,239,0.25)",background:"rgba(9,12,22,0.82)",backdropFilter:"blur(8px)",color:"#5b8def",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer"},children:[l.jsx(Cx,{size:10})," Reset"]})]}),!p&&l.jsx("div",{style:{padding:"2px 8px",borderRadius:5,fontSize:9,fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.3)",background:"rgba(9,12,22,0.6)",border:"1px solid rgba(91,141,239,0.08)",backdropFilter:"blur(4px)"},children:"Molette pour zoomer · Glisser pour naviguer"})]}),l.jsx("canvas",{ref:i,style:{width:"100%",height:"100%",borderRadius:10,cursor:p?"grab":"crosshair"},onMouseMove:v,onMouseDown:M,onMouseUp:d,onMouseLeave:m}),r&&l.jsxs("div",{style:{position:"absolute",left:r.x+12,top:r.y-42,background:"rgba(12,16,28,0.96)",border:"1px solid rgba(91,141,239,0.3)",borderRadius:8,padding:"6px 10px",pointerEvents:"none",fontFamily:"'DM Mono',monospace",fontSize:10,color:"#a0b4dc",zIndex:20},children:[l.jsxs("div",{children:["Phase: ",l.jsx("span",{style:{color:"#fff"},children:r.time.toFixed(4)})]}),l.jsxs("div",{children:["Flux:  ",l.jsx("span",{style:{color:"#fff"},children:r.flux.toFixed(6)})]})]})]})}function eh({score:t,size:e=160,scoreStd:n=null}){const[i,r]=Z.useState(0);Z.useEffect(()=>{let h;const f=performance.now(),g=x=>{const M=Math.min((x-f)/1400,1);r(t*(1-(1-M)**4)),M<1&&(h=requestAnimationFrame(g))};return h=requestAnimationFrame(g),()=>cancelAnimationFrame(h)},[t]);const s=e/2-16,o=Math.PI*s,a=o*(1-i),c=i>=.7?"#34d399":i>=.35?"#fbbf24":"#f87171",u=i>=.85?"Exoplanète très probable":i>=.7?"Exoplanète probable":i>=.55?"Candidat à confirmer":i>=.35?"Indéterminé":i>=.15?"Probable faux positif":"Faux positif très probable",p=n!=null?Math.round(n*100):null;return l.jsxs("div",{style:{display:"flex",flexDirection:"column",alignItems:"center",gap:8},children:[l.jsxs("svg",{width:e,height:e/2+22,viewBox:`0 0 ${e} ${e/2+22}`,children:[l.jsx("path",{d:`M 16 ${e/2} A ${s} ${s} 0 0 1 ${e-16} ${e/2}`,fill:"none",stroke:"rgba(91,141,239,0.1)",strokeWidth:"10",strokeLinecap:"round"}),l.jsx("path",{d:`M 16 ${e/2} A ${s} ${s} 0 0 1 ${e-16} ${e/2}`,fill:"none",stroke:c,strokeWidth:"10",strokeLinecap:"round",strokeDasharray:o,strokeDashoffset:a,style:{filter:`drop-shadow(0 0 7px ${c}50)`,transition:"stroke .05s"}}),l.jsxs("text",{x:e/2,y:e/2-8,textAnchor:"middle",fill:"#fff",fontFamily:"'DM Mono',monospace",fontSize:"28",fontWeight:"700",children:[(i*100).toFixed(1),"%"]}),l.jsx("text",{x:e/2,y:e/2+13,textAnchor:"middle",fill:"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace",fontSize:"10",children:"SCORE IA"})]}),l.jsx("div",{style:{padding:"4px 14px",borderRadius:20,fontSize:11,fontFamily:"'DM Mono',monospace",color:c,background:`${c}15`,border:`1px solid ${c}35`},children:u}),p!=null&&l.jsxs("div",{style:{fontSize:10,fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.5)",background:"rgba(91,141,239,0.05)",border:"1px solid rgba(91,141,239,0.12)",borderRadius:6,padding:"2px 10px",letterSpacing:.3},children:["IC 95% : ",(i*100).toFixed(1),"% ± ",p,"%"]})]})}function G3({data:t}){if(!t)return null;const e=t.characterization||{},n=t.metadata||{},i=e.transit_depth_ppm?e.transit_depth_ppm>5e3?"Le creux photometrique est tres marque, donc le transit est visuellement plus facile a reperer.":e.transit_depth_ppm>1e3?"Le transit est net mais pas gigantesque, ce qui correspond a un signal exploitable.":"Le transit est subtil, donc la decision depend davantage du bruit et de la stabilite du signal.":"La profondeur de transit sera interpretable apres l'analyse complete.",r=e.snr?e.snr>10?"Le signal se detache bien du bruit, ce qui rend la detection plus solide.":e.snr>5?"Le signal est present mais demande encore une lecture prudente.":"Le signal est proche du bruit, donc il faut rester prudent dans l'interpretation.":"Le niveau de confiance du signal sera estime apres calcul du SNR.",s=n.known_disposition?`Le catalogue NASA reference cette cible comme ${n.known_disposition.toLowerCase()}.`:"Aucune correspondance directe n'a ete trouvee dans le catalogue pour comparer le resultat.",o=[{label:"Rythme orbital",value:t.period_days?`${t.period_days} j`:"n/d",text:t.period_days?`La baisse de luminosite se repete environ tous les ${t.period_days} jours.`:"La periode sera visible des que le repliement de la courbe est disponible.",icon:ga,color:"#5b8def"},{label:"Profondeur du transit",value:e.transit_depth_ppm?`${e.transit_depth_ppm.toLocaleString()} ppm`:"n/d",text:i,icon:Ax,color:"#7c3aed"},{label:"Qualite du signal",value:e.snr?`SNR ${e.snr.toFixed(1)}`:"n/d",text:r,icon:Fc,color:"#fbbf24"},{label:"Comparaison catalogue",value:n.known_disposition||"Non renseigne",text:s,icon:Kp,color:"#34d399"}];return l.jsxs(Qe,{style:{padding:14},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",gap:10,marginBottom:12,flexWrap:"wrap"},children:[l.jsxs("div",{children:[l.jsx("h3",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:4,textTransform:"uppercase",letterSpacing:1.5},children:"Lecture des donnees"}),l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:"Ce que racontent les chiffres"})]}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"},children:"Interprete en langage simple"})]}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:10},children:o.map(a=>l.jsxs("div",{style:{padding:"12px 12px 10px",borderRadius:12,background:"rgba(91,141,239,0.04)",border:"1px solid rgba(91,141,239,0.08)"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,marginBottom:8},children:[l.jsx("div",{style:{width:28,height:28,borderRadius:9,display:"flex",alignItems:"center",justifyContent:"center",background:`${a.color}16`,border:`1px solid ${a.color}30`},children:l.jsx(a.icon,{size:14,style:{color:a.color}})}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.5)",textTransform:"uppercase",letterSpacing:1},children:a.label})]}),l.jsx("div",{style:{fontSize:18,fontWeight:700,color:"#e4e8f7",fontFamily:"'DM Mono',monospace",marginBottom:6},children:a.value}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.56)",lineHeight:1.55},children:a.text})]},a.label))})]})}function W3({target:t}){var c,u;const[e,n]=Z.useState(null),[i,r]=Z.useState(!1);if(Z.useEffect(()=>{t&&(n(null),r(!0),ln(`${jt}/api/star_info?target=${encodeURIComponent(t)}`).then(p=>p.ok?p.json():null).then(p=>{n(p||null),r(!1)}).catch(()=>r(!1)))},[t]),i)return l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:10,padding:"18px 0",color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace",fontSize:12},children:[l.jsx(mi,{size:14,style:{animation:"spin 1s linear infinite"}})," Recherche NASA Exoplanet Archive…"]});if(!e||!e.stellar&&!((c=e.planets)!=null&&c.length))return null;const s=e.stellar||{},o=s.distance_pc?Math.round(s.distance_pc*3.2616).toLocaleString():null,a=[s.teff&&{label:"Température",value:`${Math.round(s.teff).toLocaleString()} K`,color:"#f59e0b"},s.radius&&{label:"Rayon",value:`${s.radius.toFixed(2)} R☉`,color:"#e4e8f7"},s.mass&&{label:"Masse",value:`${s.mass.toFixed(2)} M☉`,color:"#e4e8f7"},o&&{label:"Distance",value:`${o} al`,color:"#7c3aed"},s.kmag&&{label:"Magnitude K",value:s.kmag.toFixed(2),color:"rgba(160,180,220,0.7)"}].filter(Boolean);return l.jsxs(Qe,{style:{padding:24,marginTop:0},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:20,flexWrap:"wrap",gap:8},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:10},children:[l.jsx(sa,{size:15,color:"#5b8def"}),l.jsxs("span",{style:{fontSize:13,fontWeight:700,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:["Données stellaires — ",e.hostname]})]}),l.jsx("span",{style:{fontSize:9,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"},children:e.source})]}),l.jsxs("div",{style:{display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-start"},children:[a.length>0&&l.jsxs("div",{style:{flex:1,minWidth:200},children:[l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.4,marginBottom:12},children:"Étoile hôte"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:8},children:a.map(p=>l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",padding:"8px 12px",background:"rgba(15,18,30,0.5)",borderRadius:8,border:"1px solid rgba(91,141,239,0.07)"},children:[l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"},children:p.label}),l.jsx("span",{style:{fontSize:11,fontWeight:600,color:p.color,fontFamily:"'DM Mono',monospace"},children:p.value})]},p.label))})]}),((u=e.planets)==null?void 0:u.length)>0&&l.jsxs("div",{style:{flex:2,minWidth:260},children:[l.jsxs("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.4,marginBottom:12},children:[e.planets.length," planète",e.planets.length>1?"s":""," confirmée",e.planets.length>1?"s":""]}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:6},children:e.planets.map((p,h)=>l.jsxs("div",{style:{padding:"10px 14px",background:"rgba(52,211,153,0.05)",borderRadius:8,border:"1px solid rgba(52,211,153,0.12)",display:"flex",flexWrap:"wrap",gap:12,alignItems:"center"},children:[l.jsx("span",{style:{fontSize:13,fontWeight:600,color:"#34d399",fontFamily:"'DM Mono',monospace",minWidth:110},children:p.name}),l.jsxs("div",{style:{display:"flex",gap:12,flexWrap:"wrap"},children:[p.period_days!=null&&l.jsxs("span",{style:{fontSize:11,color:"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace"},children:[p.period_days.toFixed(1)," j"]}),p.radius_earth!=null&&l.jsxs("span",{style:{fontSize:11,color:"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace"},children:[p.radius_earth.toFixed(2)," R⊕"]}),p.eq_temp!=null&&l.jsxs("span",{style:{fontSize:11,color:"#f59e0b",fontFamily:"'DM Mono',monospace"},children:[Math.round(p.eq_temp)," K"]}),p.disc_year&&l.jsx("span",{style:{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"},children:p.disc_year})]})]},h))})]})]})]})}function Lx({data:t}){if(!t)return null;const e=t.characterization,n=t.metadata;if(!e&&!n)return null;const i=[];return t.mission&&i.push({icon:Zp,label:"Mission",val:t.mission}),t.period_days&&i.push({icon:ga,label:"Période",val:`${t.period_days} j`}),t.points_count&&i.push({icon:Yp,label:"Points mesurés",val:t.points_count.toLocaleString()}),e!=null&&e.planet_type&&i.push({icon:sa,label:"Type planète",val:e.planet_type}),e!=null&&e.planet_radius_earth&&i.push({icon:$p,label:"Rayon planète",val:`${e.planet_radius_earth} R⊕`}),e!=null&&e.transit_depth_ppm&&i.push({icon:Ax,label:"Prof. transit",val:`${e.transit_depth_ppm.toLocaleString()} ppm`}),e!=null&&e.snr&&i.push({icon:Fc,label:"SNR",val:e.snr.toFixed(1)}),e!=null&&e.confidence&&i.push({icon:A3,label:"Confiance",val:e.confidence}),n!=null&&n.star_temperature_k&&i.push({icon:Nc,label:"Temp. étoile",val:`${n.star_temperature_k.toLocaleString()} K`}),n!=null&&n.star_radius_solar&&i.push({icon:sa,label:"Rayon étoile",val:`${n.star_radius_solar} R☉`}),n!=null&&n.kepler_magnitude&&i.push({icon:ur,label:"Magnitude Kepler",val:n.kepler_magnitude}),n!=null&&n.known_disposition&&i.push({icon:Kp,label:"Statut NASA",val:n.known_disposition}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6},children:i.map((r,s)=>l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"7px 10px",borderRadius:8,background:"rgba(91,141,239,0.04)",border:"1px solid rgba(91,141,239,0.08)"},children:[l.jsx(r.icon,{size:12,style:{color:"rgba(91,141,239,0.5)",flexShrink:0}}),l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.45)",textTransform:"uppercase",letterSpacing:1},children:r.label}),l.jsx("div",{style:{fontSize:12,color:"#e4e8f7",marginTop:1},children:r.val??"—"})]})]},s))})}const M0={koi_period:"Période orbitale",koi_period_err1:"Incertitude période",koi_period_err2:"Incertitude période",koi_time0bk:"Époque du transit",koi_time0bk_err1:"Incertitude époque",koi_time0bk_err2:"Incertitude époque",koi_impact:"Paramètre d'impact",koi_impact_err1:"Incertitude impact",koi_impact_err2:"Incertitude impact",koi_duration:"Durée du transit",koi_duration_err1:"Incertitude durée",koi_duration_err2:"Incertitude durée",koi_depth:"Profondeur du transit",koi_depth_err1:"Incertitude profondeur",koi_depth_err2:"Incertitude profondeur",koi_prad:"Rayon de la planète",koi_prad_err1:"Incertitude rayon planète",koi_prad_err2:"Incertitude rayon planète",koi_teq:"Température d'équilibre",koi_insol:"Flux d'irradiation",koi_insol_err1:"Incertitude flux",koi_insol_err2:"Incertitude flux",koi_steff:"Température de l'étoile",koi_steff_err1:"Incertitude temp. étoile",koi_steff_err2:"Incertitude temp. étoile",koi_slogg:"Gravité de surface étoile",koi_slogg_err1:"Incertitude gravité",koi_slogg_err2:"Incertitude gravité",koi_srad:"Rayon de l'étoile",koi_srad_err1:"Incertitude rayon étoile",koi_srad_err2:"Incertitude rayon étoile",koi_model_snr:"Rapport signal / bruit",koi_fpflag_nt:"Flag non-transit",koi_fpflag_ss:"Flag étoile secondaire",koi_fpflag_co:"Flag contamination",koi_fpflag_ec:"Flag éphéméride",is_tess:"Mission TESS",glon:"Longitude galactique",glat:"Latitude galactique",koi_kepmag:"Magnitude Kepler/TESS"},Cr={bls_period:"Période orbitale en jours détectée par BLS (Box Least Squares). L'algorithme balaye la courbe de lumière sur une grille de périodes et sélectionne celle qui maximise le rapport signal/bruit du creux en forme de boîte. Utilisé pour Kepler et TESS.",bls_duration:"Durée du transit en jours telle qu'estimée par BLS. Correspond au temps pendant lequel la planète occulte partiellement l'étoile, du premier au dernier contact. Dépend du rayon orbital, de la taille de l'étoile et de l'inclinaison.",bls_depth:"Profondeur du transit en flux relatif (ou ppm). Rapport de la surface occultée sur la surface totale de l'étoile : δ = (Rp/R★)². Terre devant le Soleil ≈ 84 ppm, Jupiter ≈ 10 000 ppm. Feature clé pour estimer la taille de la planète.",bls_depth_ppm:"Profondeur du transit exprimée en parties par million (ppm). Même grandeur que bls_depth mais normalisée pour faciliter la comparaison entre étoiles de luminosités différentes.",bls_snr:"Rapport Signal/Bruit (SNR) du meilleur signal BLS. SNR = profondeur / bruit RMS × √(nb transits observés). Un SNR > 7 est généralement requis pour considérer un signal comme détectable. En dessous de 5 : résultat peu fiable.",bls_score:"Score de qualité BLS normalisé entre 0 et 1. Combine le SNR et la cohérence de la forme du signal. Plus le score est proche de 1, plus le signal ressemble à un transit planétaire propre.",bls_t0:"Époque du premier transit (T0) en jours BJD. Temps du centre du premier transit observé. Avec la période, permet de prédire tous les transits futurs et passés pour valider la périodicité.",koi_period:"Période orbitale en jours (catalogue KOI Kepler). Estimée par BLS sur les données PDC-SAP de Kepler, puis raffinée par modélisation de transit. Précision typique : < 0.001 jour.",koi_time0bk:"Époque du premier transit en BKJD (Barycentric Kepler Julian Date). Instant exact du centre du premier transit observé par Kepler. Permet de propager l'éphéméride sur toute la mission.",koi_impact:"Paramètre d'impact b ∈ [0, 1+Rp/R★]. Distance entre le centre de l'étoile et la trajectoire de la planète en unités de rayon stellaire. b=0 : transit central, b→1 : transit rasant. Influe fortement sur la forme et la durée du creux.",koi_duration:"Durée totale du transit en heures (Kepler). Du premier au dernier contact externe. Dépend du rayon orbital (loi de Kepler), de l'inclinaison et des rayons des deux corps.",koi_depth:"Profondeur du transit Kepler en ppm. Ratio de la surface planétaire sur la surface stellaire : δ = (Rp/R★)². Directement utilisé pour estimer le rayon de la planète via koi_prad.",koi_model_snr:"SNR du modèle de transit ajusté par DV (Data Validation) de Kepler. Plus fiable que le SNR BLS car basé sur un modèle Mandel-Agol ajusté. Valeur < 7 : signal ambigu, > 15 : signal robuste.",koi_prad:"Rayon estimé de la planète en rayons terrestres (R⊕). Calculé via : Rp = R★ × √δ. La précision dépend directement de la connaissance du rayon stellaire. Super-Terres : 1–2 R⊕, Neptunes : 2–6 R⊕, Jupiters : > 10 R⊕.",koi_teq:"Température d'équilibre planétaire en Kelvin. Estimée par : Teq = T★ × (R★/2a)^0.5 × (1-A)^0.25, avec A=0.3 (albédo Bond). Zone habitable approximative : 200–300 K selon le type stellaire.",koi_insol:"Irradiation reçue par la planète en flux terrestres (S⊕). S = L★ / (4πa²) normalisé par la valeur terrestre. Zone habitable optimiste : 0.25 < S < 11 S⊕. Feature importante pour la classification habitabilité.",koi_steff:"Température effective de l'étoile hôte en Kelvin. Déterminée par spectroscopie ou photométrie SED. Classification : M < 3 900 K, K 3 900–5 200 K, G 5 200–6 000 K (Soleil : 5 778 K), F 6 000–7 500 K. Influe sur le contraste du transit.",koi_slogg:"Gravité de surface stellaire log g (cm/s²). log g = log(GM★/R★²). Naines principales : 4.0–4.8, Sous-géantes : 3.5–4.0, Géantes : < 3.5. Utile pour rejeter les faux positifs liés aux géantes contaminantes.",koi_srad:"Rayon de l'étoile hôte en rayons solaires (R☉). Déduit de la luminosité et de T_eff via L = 4πR²σT⁴. Paramètre crucial : une erreur sur R★ se répercute directement sur l'estimation de Rp.",star_teff:"Température effective de l'étoile (TESS/TIC). Équivalent de koi_steff pour les cibles TESS. Issu du catalogue TIC v8 (TESS Input Catalog), dérivé de la photométrie multi-bande ou de Gaia.",star_logg:"Gravité de surface de l'étoile (TESS/TIC). Même interprétation que koi_slogg. Permet de distinguer les naines M (cibles privilégiées de TESS pour les super-Terres) des géantes.",star_radius:"Rayon stellaire en R☉ (TESS/TIC). Estimé à partir de Gaia DR3 (parallaxe + luminosité) ou du catalogue TIC. Précision typique : 5–10% pour les étoiles brillantes de TESS.",koi_fpflag_nt:"Flag 'Non-Transit Shape' : 1 si la forme du signal ne correspond pas à un transit planétaire en boîte (trapèze asymétrique, pentes incompatibles). Signature typique d'une étoile binaire éclipsante ou d'un artefact instrumental.",koi_fpflag_ss:"Flag 'Significant Secondary' : 1 si un transit secondaire significatif est détecté à mi-période. Un transit planétaire n'a pas de transit secondaire (la planète n'émet pas de lumière propre). Signature forte d'une binaire éclipsante.",koi_fpflag_co:"Flag 'Centroid Offset' : 1 si le centroïde du flux se déplace pendant le transit, indiquant que la source est une étoile voisine dans le pixel Kepler (~4 arcsec) et non la cible. Test de contamination critique pour Kepler.",koi_fpflag_ec:"Flag 'Ephemeris Contamination' : 1 si la période du signal correspond à une éclipse connue dans un rayon de 2 arcsec. La longue PSF de Kepler peut contaminer une cible avec une binaire brillante voisine.",koi_kepmag:"Magnitude Kepler de l'étoile (bande 430–890 nm). Kp < 12 : étoile brillante, SNR élevé, bonne détection. Kp > 15 : étoile faible, bruit de photon dominant. Feature indirecte de la qualité de la courbe de lumière.",glon:"Longitude galactique en degrés (0–360°). Position dans le plan de la Voie Lactée. Régions proches du centre galactique (glon ≈ 0°) ont une densité stellaire plus élevée, augmentant le risque de contamination.",glat:"Latitude galactique en degrés (-90° à +90°). Hauteur au-dessus du plan galactique. glat élevée → moins d'étoiles de fond → moins de faux positifs par contamination. Le champ principal Kepler est à glat ≈ +13°.",mean:"Moyenne du flux sur la courbe repliée (phase-folded). Valeur nominale ≈ 1.0 après normalisation. Un écart significatif indique une dérive résiduelle ou une étoile variable non corrigée par le flattening.",variance:"Variance du flux : dispersion quadratique autour de la moyenne. Forte variance → bruit élevé ou variabilité stellaire intrinsèque (pulsations, taches). Feature de qualité du signal brut.",skewness:"Asymétrie (3ème moment) de la distribution de flux. Un transit crée une queue négative : skewness légèrement négatif. Une forte asymétrie positive peut trahir une étoile variable ou un faux positif.",kurtosis:"Kurtosis (4ème moment) : aplatissement de la distribution. Un transit net génère un pic de kurtosis positif (distribution leptokurtique) car la majorité du flux est concentrée autour de 1.0 avec quelques points très bas.",abs_energy:"Énergie absolue : Σ(flux²). Proportionnelle à la puissance totale du signal. Permet de distinguer les étoiles calmes (abs_energy ≈ N × 1.0) des variables (abs_energy élevée).",mean_abs_change:"Variation absolue moyenne entre points consécutifs : mean(|f[i+1] - f[i]|). Mesure la rugosité locale de la courbe. Un transit propre a des flancs raides (forte variation) mais une base et un plateau plats.",count_above_mean:"Nombre de points au-dessus de la moyenne. Dans un signal transit propre : la grande majorité (>90%) des points est à flux ≈ 1.0, au-dessus de la moyenne légèrement abaissée par le creux.",count_below_mean:"Nombre de points sous la moyenne. Directement proportionnel au nombre de points dans le creux de transit. Un transit étroit (planète petite) donne peu de points sous la moyenne.",longest_strike_below_mean:"Longueur de la plus longue séquence continue sous la moyenne. Dans la courbe repliée, correspond à la durée du transit en nombre de points. Feature très discriminante pour séparer transits (séquence courte et nette) du bruit.",longest_strike_above_mean:"Longueur de la plus longue séquence continue au-dessus de la moyenne. Complémentaire : dans un signal transit, presque toute la courbe est au-dessus sauf le creux. Un bruit aléatoire donne des séquences plus courtes.",sum_of_reoccurring_values:"Somme des valeurs de flux qui se répètent exactement. Indicateur indirect de quantification ou de répétabilité du signal. Un transit périodique produit des flux similaires à chaque répétition.",ratio_beyond_r_sigma:"Fraction des points à plus de r×σ de la moyenne (r typiquement 1 ou 2). Détecte les outliers et les pics extrêmes. Un transit planétaire produit des points clairement sous -2σ, mais peu de points au-delà de +2σ.",autocorrelation:"Autocorrélation du flux à un décalage temporel (lag) donné. Mesure la similitude du signal avec lui-même décalé. Un transit périodique produit des pics d'autocorrélation aux multiples de la période, signature d'un signal régulier.",fft_coefficient:"Coefficient complexe de la FFT (Fast Fourier Transform) à une fréquence donnée. Décompose le signal en composantes sinusoïdales. Un transit présente des harmoniques nettes à f=1/P, 2/P, 3/P... reflétant sa forme en créneau.",cwt_coefficients:"Coefficients de la transformée en ondelettes continues (CWT, Morlet). Analyse temps-fréquence multi-résolution. Localise précisément les transitoires dans le temps ET la fréquence. Robuste aux variations d'amplitude entre transits.",agg_linear_trend:"Pente de la tendance linéaire sur un segment agrégé de la courbe. Détecte les dérives lentes non corrigées par le flattening (ex : activité stellaire résiduelle, systematic du satellite). Doit être proche de zéro pour un bon signal.",binned_entropy:"Entropie de Shannon après discrétisation du flux en bins. Mesure le désordre statistique de la distribution. Transit propre → distribution bimodale (plateau + creux) → entropie réduite. Bruit gaussien → entropie maximale.",permutation_entropy:"Entropie de permutation : complexité des patterns locaux d'ordonnancement sur fenêtres glissantes. Très sensible aux structures cachées. Un transit crée des patterns d'ordre répétés (montée, plateau, descente) détectables même en présence de bruit.",sample_entropy:"Entropie d'échantillon : probabilité que deux séquences de longueur m restent similaires en passant à m+1. Faible pour un signal régulier et périodique (transit), élevée pour un bruit aléatoire. Robuste aux artefacts de longueur finie."};function Fx(t){const e=(t||"").replace("flux__","").replace("sci_","");if(Cr[e])return Cr[e];if(Cr[t])return Cr[t];const n=e.replace(/_err[12]$/,"");if(n!==e&&Cr[n])return Cr[n];const i=e.split("__")[0];return Cr[i]||null}function fc(t){const e=(t||"").replace("flux__","").replace("sci_","");return M0[e]||M0[t]||e}function X3({features:t}){if(!(t!=null&&t.length))return null;const e=Math.max(...t.map(r=>r.weight||r.importance||0)),[n,i]=Z.useState(null);return l.jsxs("div",{style:{position:"relative"},children:[l.jsx("h4",{style:{fontSize:10,color:"rgba(160,180,220,0.5)",marginBottom:8,textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"},children:"Top features (interprétabilité)"}),t.map((r,s)=>{const o=r.weight??r.importance??0,a=o/e*100,c=fc(r.name),u=Fx(r.name);return l.jsxs("div",{style:{position:"relative",marginBottom:5},onMouseEnter:u?p=>{const h=p.currentTarget.getBoundingClientRect();i({text:u,rawName:r.name,x:h.left,y:h.bottom+6})}:void 0,onMouseLeave:()=>i(null),children:[l.jsx("div",{style:{position:"absolute",left:0,top:0,bottom:0,width:`${a}%`,background:"rgba(91,141,239,0.08)",borderRadius:6,transition:"width .4s"}}),l.jsxs("div",{style:{position:"relative",display:"flex",justifyContent:"space-between",alignItems:"center",padding:"5px 8px",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:u?"help":"default"},children:[l.jsx("span",{style:{color:"rgba(160,180,220,0.7)",maxWidth:"75%",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},children:c}),l.jsxs("span",{style:{color:"#5b8def"},children:[(o*100).toFixed(1),"%"]})]})]},s)}),n&&da.createPortal(l.jsxs("div",{style:{position:"fixed",left:Math.min(n.x,window.innerWidth-340),top:n.y,width:320,background:"rgba(6,9,20,0.97)",border:"1px solid rgba(91,141,239,0.3)",borderRadius:10,padding:"12px 16px",zIndex:99999,boxShadow:"0 12px 40px rgba(0,0,0,0.7)",pointerEvents:"none"},children:[l.jsx("div",{style:{fontSize:11,fontWeight:700,color:"#5b8def",fontFamily:"'DM Mono',monospace",marginBottom:6},children:fc(n.rawName)}),l.jsx("div",{style:{fontSize:11,color:"rgba(200,215,240,0.75)",fontFamily:"'Space Grotesk',sans-serif",lineHeight:1.7},children:n.text})]}),document.body)]})}function q3({status:t}){if(!t)return null;const e=[{l:"Backend",ok:t.status==="online"},{l:"IA",ok:t.ai_loaded},{l:"Catalog",ok:t.catalog_loaded}];return l.jsx("div",{style:{display:"flex",gap:5},children:e.map((n,i)=>l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:3,padding:"3px 8px",borderRadius:6,fontSize:10,fontFamily:"'DM Mono',monospace",background:n.ok?"rgba(52,211,153,0.06)":"rgba(248,113,113,0.06)",border:`1px solid ${n.ok?"rgba(52,211,153,0.15)":"rgba(248,113,113,0.15)"}`,color:n.ok?"#34d399":"#f87171"},children:[n.ok?l.jsx(ma,{size:9}):l.jsx(hr,{size:9})," ",n.l]},i))})}function $3({stat:t}){const[e,n]=Z.useState(!1),[i,r]=Z.useState({left:0,top:0}),s=Z.useRef(null),o=272,a=()=>{if(!s.current)return;const u=s.current.getBoundingClientRect(),p=u.left+u.width/2,h=u.top,f=Math.max(8,Math.min(window.innerWidth-o-8,p-o/2)),g=h>130?u.top-8:u.bottom+8,x=h>130;r({left:f,top:g,above:x}),n(!0)},c=()=>n(!1);return l.jsxs("div",{ref:s,style:{position:"relative"},onMouseEnter:a,onMouseLeave:c,onFocus:a,onBlur:c,tabIndex:0,children:[l.jsxs(Qe,{style:{padding:"14px 16px",textAlign:"center",cursor:"help"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",gap:6,marginBottom:2},children:[l.jsx("div",{style:{fontSize:11,color:"#e4e8f7"},children:t.label}),l.jsx(Lc,{size:11,style:{color:"rgba(91,141,239,0.6)"}})]}),l.jsx("div",{style:{fontSize:22,fontWeight:700,fontFamily:"'DM Mono',monospace",color:"#5b8def",marginBottom:2},children:t.val}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)"},children:t.sub})]}),e&&l.jsx("div",{style:{position:"fixed",left:i.left,...i.above?{bottom:window.innerHeight-i.top+6}:{top:i.top},width:o,padding:"10px 13px",borderRadius:10,background:"rgba(8,12,22,0.97)",border:"1px solid rgba(91,141,239,0.28)",color:"rgba(224,232,245,0.88)",fontSize:10.5,lineHeight:1.62,textAlign:"left",zIndex:9999,boxShadow:"0 16px 36px rgba(0,0,0,0.5)",pointerEvents:"none",backdropFilter:"blur(8px)"},children:t.hint})]})}function K3(){const[t,e]=Z.useState(!1),[n,i]=Z.useState({left:0,top:0}),r=Z.useRef(null),s=400,o=()=>{e(!0)};return l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:7,marginBottom:14,position:"relative"},children:[l.jsx("h3",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",margin:0,textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"},children:"Importance des features"}),l.jsx("div",{ref:r,onMouseEnter:o,onMouseLeave:()=>e(!1),style:{display:"flex",alignItems:"center",justifyContent:"center",width:17,height:17,borderRadius:"50%",cursor:"help",flexShrink:0,background:"rgba(91,141,239,0.12)",border:"1px solid rgba(91,141,239,0.3)",color:"#5b8def",fontSize:10,fontWeight:700,fontFamily:"'DM Mono',monospace",transition:"background .2s",userSelect:"none"},onMouseEnter2:a=>{a.currentTarget.style.background="rgba(91,141,239,0.22)",o()},children:l.jsx(Lc,{size:10})}),t&&da.createPortal(l.jsxs("div",{style:{position:"fixed",top:"50%",left:"50%",transform:"translate(-50%, -50%)",width:s,maxWidth:"calc(100vw - 32px)",padding:"18px 20px",borderRadius:14,background:"rgba(7,10,20,0.98)",border:"1px solid rgba(91,141,239,0.32)",zIndex:2147483647,boxShadow:"0 24px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(91,141,239,0.1)",pointerEvents:"none",backdropFilter:"blur(14px)",fontFamily:"'DM Mono',monospace",animation:"fadeIn .18s ease-out"},children:[l.jsx("div",{style:{fontSize:11,fontWeight:700,color:"#5b8def",marginBottom:10,textTransform:"uppercase",letterSpacing:1.2},children:"Qu'est-ce que l'importance des features ?"}),l.jsx("div",{style:{fontSize:10.5,color:"rgba(224,232,245,0.82)",lineHeight:1.65,marginBottom:12},children:"Chaque feature est une variable numerique calculee sur la courbe de lumiere (periode, rayon, profondeur du transit…). Le modele XGBoost assigne a chacune un score d'importance qui mesure combien elle contribue aux bonnes predictions."}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.55)",marginBottom:8,textTransform:"uppercase",letterSpacing:1},children:"Pourquoi certaines sont plus utiles ?"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:7},children:[{icon:"📐",label:"Taille physique",text:"Le rayon de la planete (koi_prad) est le signal le plus fort : une grande planete occulte plus de lumiere, rendant le transit facilement distinguable du bruit."},{icon:"⏱",label:"Geometrie temporelle",text:"Le rapport duree/periode (duty_cycle) revele si le transit est trop court ou trop long par rapport a l'orbite — un faux positif comme une etoile binaire a souvent un duty_cycle anormal."},{icon:"📡",label:"Qualite du signal",text:"Le SNR proxy et la temperature de l'etoile (koi_steff) permettent de savoir si le signal emerge suffisamment du bruit photometrique, indispensable pour valider une detection."},{icon:"🔗",label:"Coherence physique",text:"Le ratio rayon planete / rayon etoile (ratio_prad_srad) verifie que la geometrie est coherente : une planete plus grande que son etoile est physiquement impossible et trahit un faux positif."}].map((a,c)=>l.jsxs("div",{style:{display:"flex",gap:9,padding:"7px 9px",borderRadius:8,background:"rgba(91,141,239,0.04)",border:"1px solid rgba(91,141,239,0.08)"},children:[l.jsx("span",{style:{fontSize:14,flexShrink:0},children:a.icon}),l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:9.5,fontWeight:600,color:"#5b8def",marginBottom:2,textTransform:"uppercase",letterSpacing:.8},children:a.label}),l.jsx("div",{style:{fontSize:10,color:"rgba(200,215,240,0.75)",lineHeight:1.55},children:a.text})]})]},c))}),l.jsx("div",{style:{marginTop:10,fontSize:9.5,color:"rgba(160,180,220,0.38)",lineHeight:1.5},children:"Le modele apprend seul quelles variables separent le mieux planetes confirmeees et faux positifs sur le catalogue KOI de la mission Kepler."})]}),document.body)]})}function Y3(){const[t,e]=Z.useState(!1),n=Z.useRef(null),i=420,r=()=>e(!0),s=()=>e(!1);return l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:7,marginBottom:14},children:[l.jsx("h3",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",margin:0,textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"},children:"Matrice de confusion"}),l.jsx("div",{ref:n,onMouseEnter:r,onMouseLeave:s,style:{display:"flex",alignItems:"center",justifyContent:"center",width:17,height:17,borderRadius:"50%",cursor:"help",flexShrink:0,background:"rgba(91,141,239,0.12)",border:"1px solid rgba(91,141,239,0.3)",color:"#5b8def",transition:"background .2s",userSelect:"none"},children:l.jsx(Lc,{size:10})}),t&&da.createPortal(l.jsxs("div",{style:{position:"fixed",top:"50%",left:"50%",transform:"translate(-50%, -50%)",width:i,maxWidth:"calc(100vw - 32px)",padding:"18px 20px",borderRadius:14,background:"rgba(7,10,20,0.98)",border:"1px solid rgba(91,141,239,0.32)",zIndex:2147483647,boxShadow:"0 24px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(91,141,239,0.1)",pointerEvents:"none",backdropFilter:"blur(14px)",fontFamily:"'DM Mono',monospace",animation:"fadeIn .18s ease-out"},children:[l.jsx("div",{style:{fontSize:11,fontWeight:700,color:"#5b8def",marginBottom:10,textTransform:"uppercase",letterSpacing:1.2},children:"Qu'est-ce que la matrice de confusion ?"}),l.jsx("div",{style:{fontSize:10.5,color:"rgba(224,232,245,0.82)",lineHeight:1.65,marginBottom:14},children:"La matrice de confusion compare les predictions du modele aux vraies etiquettes du jeu de test. Elle se divise en 4 cellules selon que la prediction est correcte ou non."}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:14},children:[{label:"Vrai Negatif (TN)",color:"#34d399",text:"L'etoile n'a pas d'exoplanete et le modele le dit correctement. Pas de transit — bonne detection.",icon:"✅"},{label:"Faux Positif (FP)",color:"#f87171",text:"Le modele croit detecter une exoplanete, mais c'est une erreur (binaire a eclipse, bruit stellaire…). Alarme injustifiee.",icon:"⚠️"},{label:"Faux Negatif (FN)",color:"#f87171",text:"Une vraie exoplanete existe, mais le modele l'a ratee. C'est la pire erreur pour la recherche : on passe a cote d'une decouverte.",icon:"❌"},{label:"Vrai Positif (TP)",color:"#34d399",text:"Une vraie exoplanete est correctement detectee. C'est le resultat ideal que l'on cherche a maximiser.",icon:"🌍"}].map((o,a)=>l.jsxs("div",{style:{padding:"9px 11px",borderRadius:9,background:`rgba(${o.color==="#34d399"?"74,222,128":"248,113,113"},0.06)`,border:`1px solid ${o.color}30`},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:6,marginBottom:5},children:[l.jsx("span",{style:{fontSize:13},children:o.icon}),l.jsx("span",{style:{fontSize:9.5,fontWeight:700,color:o.color,textTransform:"uppercase",letterSpacing:.8},children:o.label})]}),l.jsx("div",{style:{fontSize:10,color:"rgba(200,215,240,0.75)",lineHeight:1.55},children:o.text})]},a))}),l.jsxs("div",{style:{padding:"9px 11px",borderRadius:9,background:"rgba(91,141,239,0.05)",border:"1px solid rgba(91,141,239,0.12)",marginBottom:10},children:[l.jsx("div",{style:{fontSize:9.5,fontWeight:700,color:"#5b8def",marginBottom:5,textTransform:"uppercase",letterSpacing:.8},children:"Ce que le modele optimise"}),l.jsxs("div",{style:{fontSize:10,color:"rgba(200,215,240,0.72)",lineHeight:1.6},children:["Un modele parfait aurait ",l.jsx("span",{style:{color:"#34d399"},children:"0 FP"})," et ",l.jsx("span",{style:{color:"#34d399"},children:"0 FN"}),". En pratique, on cherche un equilibre : trop de FP = beaucoup de fausses alertes a verifier, trop de FN = on rate des exoplanetes reelles. Le F1-Score mesure cet equilibre."]})]}),l.jsx("div",{style:{fontSize:9.5,color:"rgba(160,180,220,0.38)",lineHeight:1.5},children:"Les valeurs affichees sont calculees sur le jeu de test (donnees que le modele n'a jamais vues pendant l'entrainement)."})]}),document.body)]})}function b0({features:t}){var o;if(!(t!=null&&t.length))return null;const e=((o=t[0])==null?void 0:o.importance)||1,n=["#fbbf24","#94a3b8","#cd7c3a","#5b8def","#5b8def","#5b8def","#5b8def","#5b8def"],i=["linear-gradient(90deg,#fbbf24,#f59e0b)","linear-gradient(90deg,#94a3b8,#64748b)","linear-gradient(90deg,#cd7c3a,#b45309)","linear-gradient(90deg,#5b8def,#7c3aed)"],[r,s]=Z.useState(null);return l.jsxs("div",{style:{position:"relative"},children:[t.map((a,c)=>{const u=fc(a.name),p=Fx(a.name);return l.jsxs("div",{style:{marginBottom:8,cursor:p?"help":"default"},onMouseEnter:p?h=>{const f=h.currentTarget.getBoundingClientRect();s({text:p,rawName:a.name,x:f.left,y:f.bottom+6})}:void 0,onMouseLeave:()=>s(null),children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:3,gap:8},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:6,minWidth:0},children:[l.jsxs("span",{style:{fontSize:8,fontWeight:700,fontFamily:"'DM Mono',monospace",color:n[c],flexShrink:0,width:16,textAlign:"center"},children:["#",c+1]}),l.jsx("span",{style:{color:"rgba(200,215,240,0.8)",fontSize:10,fontFamily:"'DM Mono',monospace",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},children:u})]}),l.jsxs("span",{style:{color:n[c],fontSize:10,fontFamily:"'DM Mono',monospace",fontWeight:600,flexShrink:0},children:[(a.importance*100).toFixed(1),"%"]})]}),l.jsx("div",{style:{height:5,borderRadius:3,background:"rgba(91,141,239,0.07)"},children:l.jsx("div",{style:{height:"100%",width:`${a.importance/e*100}%`,background:i[Math.min(c,i.length-1)],borderRadius:3,transition:"width .6s ease"}})})]},c)}),r&&da.createPortal(l.jsxs("div",{style:{position:"fixed",left:Math.min(r.x,window.innerWidth-340),top:r.y,width:320,background:"rgba(6,9,20,0.97)",border:"1px solid rgba(91,141,239,0.3)",borderRadius:10,padding:"12px 16px",zIndex:99999,boxShadow:"0 12px 40px rgba(0,0,0,0.7)",pointerEvents:"none"},children:[l.jsx("div",{style:{fontSize:11,fontWeight:700,color:"#5b8def",fontFamily:"'DM Mono',monospace",marginBottom:6},children:fc(r.rawName)}),l.jsx("div",{style:{fontSize:11,color:"rgba(200,215,240,0.75)",fontFamily:"'Space Grotesk',sans-serif",lineHeight:1.7},children:r.text})]}),document.body)]})}function Z3(){const t=Z.useContext(xa),[e,n]=Z.useState(null),[i,r]=Z.useState(!0),[s,o]=Z.useState(null);if(Z.useEffect(()=>{ln(`${jt}/api/metrics`).then(x=>x.json()).then(n).catch(x=>o(x.message)).finally(()=>r(!1))},[]),i)return l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",height:300,gap:10,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace"},children:[l.jsx(mi,{size:20,style:{animation:"spin 1s linear infinite"}})," Chargement des metriques..."]});if(s)return l.jsxs("div",{style:{padding:24,color:"#f87171",fontFamily:"'DM Mono',monospace",fontSize:13},children:[l.jsx(hr,{size:16,style:{marginRight:8}}),s]});if(!e)return null;if(t){const x=e.test_accuracy?Math.round(e.test_accuracy*100):null,M=e.test_precision?Math.round(e.test_precision*100):null,v=e.test_recall?Math.round(e.test_recall*100):null,d=(e.n_train||0)+(e.n_test||0),m=x?Math.round(x/10):null,_=(e.top_features||[]).slice(0,5);return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"},children:[l.jsx("h2",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:15,fontWeight:700,color:"#e4e8f7",marginBottom:0},children:"🤖 Notre intelligence artificielle"}),l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(180px,1fr))",gap:12},children:[l.jsxs(Qe,{style:{padding:"18px 20px",textAlign:"center"},children:[l.jsx("div",{style:{fontSize:34,marginBottom:6},children:"🎯"}),l.jsx("div",{style:{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#34d399"},children:x!=null?`${x}%`:"—"}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4},children:"de bonnes réponses"})]}),l.jsxs(Qe,{style:{padding:"18px 20px",textAlign:"center"},children:[l.jsx("div",{style:{fontSize:34,marginBottom:6},children:"⭐"}),l.jsx("div",{style:{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#5b8def"},children:d>0?d.toLocaleString():"—"}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4},children:"étoiles analysées pour s'entraîner"})]}),l.jsxs(Qe,{style:{padding:"18px 20px",textAlign:"center"},children:[l.jsx("div",{style:{fontSize:34,marginBottom:6},children:"✅"}),l.jsx("div",{style:{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#34d399"},children:m!=null?`${m}/10`:"—"}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4},children:"bons résultats sur 10 analyses"})]})]}),l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12},children:[l.jsxs(Qe,{style:{padding:"16px 20px"},children:[l.jsx("div",{style:{fontSize:20,marginBottom:6},children:"🔭"}),l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7",marginBottom:6},children:`Quand l'IA dit "planète"…`}),l.jsxs("p",{style:{fontSize:12,color:"rgba(200,215,240,0.65)",lineHeight:1.7,margin:0},children:["Elle a raison ",l.jsx("strong",{style:{color:"#34d399"},children:M!=null?`${M}%`:"—"})," du temps.",M!=null&&M>=80?" Très peu de fausses alertes !":M!=null&&M>=60?" Assez fiable.":""]})]}),l.jsxs(Qe,{style:{padding:"16px 20px"},children:[l.jsx("div",{style:{fontSize:20,marginBottom:6},children:"🌍"}),l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7",marginBottom:6},children:"Planètes réelles trouvées"}),l.jsxs("p",{style:{fontSize:12,color:"rgba(200,215,240,0.65)",lineHeight:1.7,margin:0},children:["Sur 10 vraies planètes, l'IA en détecte ",l.jsx("strong",{style:{color:"#34d399"},children:v!=null?`${Math.round(v/10)}`:"—"})," et en rate ",l.jsx("strong",{style:{color:"#fbbf24"},children:v!=null?`${10-Math.round(v/10)}`:"—"}),"."]})]})]}),_.length>0&&l.jsxs(Qe,{style:{padding:"16px 20px"},children:[l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7",marginBottom:12},children:"🔍 Ce que l'IA observe en priorité"}),l.jsx(b0,{features:_})]})]})}const a=e.confusion_matrix||[[0,0],[0,0]],[c,u]=[a[0][0],a[0][1]],[p,h]=[a[1][0],a[1][1]],f=Math.max(c,u,p,h)||1,g=[{label:"Precision",val:`${(e.test_precision*100).toFixed(1)}%`,sub:"test set",hint:"Parmi toutes les detections positives du modele, c'est la part qui correspond reellement a des exoplanetes. Plus elle est haute, moins le modele genere de faux positifs."},{label:"Recall",val:`${(e.test_recall*100).toFixed(1)}%`,sub:"test set",hint:"Parmi toutes les vraies exoplanetes presentes dans le jeu de test, c'est la part retrouvee par le modele. Plus il est haut, moins on manque de vraies cibles interessantes."},{label:"F1-Score",val:`${(e.test_f1*100).toFixed(1)}%`,sub:"test set",hint:"Le F1-Score combine precision et recall en une seule mesure. Il est utile quand on veut un bon compromis entre peu de faux positifs et peu de faux negatifs."},{label:"AUC-ROC",val:e.test_auc_roc.toFixed(3),sub:"test set",hint:"Cette mesure indique a quel point le modele separe bien les classes positives et negatives, quel que soit le seuil choisi. Plus on se rapproche de 1, meilleure est la separation."},{label:"CV Accuracy",val:`${(e.cv_accuracy_mean*100).toFixed(1)} +/- ${(e.cv_accuracy_std*100).toFixed(1)}%`,sub:"5-fold",hint:"Accuracy moyenne obtenue sur plusieurs decoupages du dataset. L'ecart type montre si la performance reste stable d'un fold a l'autre."},{label:"CV F1",val:`${(e.cv_f1_mean*100).toFixed(1)} +/- ${(e.cv_f1_std*100).toFixed(1)}%`,sub:"5-fold",hint:"Version cross-validation du F1-Score. Elle aide a voir si l'equilibre precision et recall reste coherent quand on change d'echantillon d'entrainement et de validation."},{label:"Features select.",val:e.n_features_selected,sub:`/ ${e.n_features_total} total`,hint:"Nombre de variables finalement retenues par le modele. Moins de features peut rendre le systeme plus lisible et parfois plus robuste si les variables ecartent le bruit inutile."},{label:"Dataset train",val:e.train_size,sub:`test: ${e.test_size}`,hint:"Taille des donnees utilisees pour entrainer et evaluer le modele. Ce contexte aide a juger si les scores reposent sur un volume de donnees plutot limite ou deja representatif."}];return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:18,animation:"fadeIn .5s ease-out"},children:[l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.46)",fontFamily:"'DM Mono',monospace"},children:"Survolez une carte pour voir ce que chaque metrique signifie."}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))",gap:10},children:g.map((x,M)=>l.jsx($3,{stat:x},M))}),l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16},children:[l.jsxs(Qe,{children:[l.jsx(Y3,{}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,maxWidth:260,margin:"0 auto"},children:[{v:c,l:"Vrais Negatifs",c:"#34d399"},{v:u,l:"Faux Positifs",c:"#f87171"},{v:p,l:"Faux Negatifs",c:"#f87171"},{v:h,l:"Vrais Positifs",c:"#34d399"}].map((x,M)=>l.jsxs("div",{style:{padding:"14px 10px",borderRadius:10,textAlign:"center",background:`rgba(${x.c==="#34d399"?"74,222,160":"248,113,113"},${.05+x.v/f*.15})`,border:`1px solid ${x.c}25`},children:[l.jsx("div",{style:{fontSize:28,fontWeight:700,fontFamily:"'DM Mono',monospace",color:x.c},children:x.v}),l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.5)",marginTop:2},children:x.l})]},M))}),l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginTop:10,fontSize:10,fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.4)",textAlign:"center"},children:[l.jsx("div",{children:"Predit : Negatif"}),l.jsx("div",{children:"Predit : Positif"})]})]}),l.jsxs(Qe,{children:[l.jsx(K3,{}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace",marginBottom:10},children:"Plus la barre est longue, plus cette variable a influencé les décisions du modèle. Survolez une ligne pour en savoir plus."}),l.jsx(b0,{features:(e.top_features||[]).slice(0,8)})]})]}),l.jsxs(Qe,{children:[l.jsx("h3",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:14,textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"},children:"Performance cross-validation (5 folds)"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:10},children:[{label:"Accuracy",val:e.cv_accuracy_mean,std:e.cv_accuracy_std,col:"#5b8def"},{label:"F1-Score",val:e.cv_f1_mean,std:e.cv_f1_std,col:"#7c3aed"},{label:"AUC-ROC",val:e.cv_auc_mean,std:e.cv_auc_std,col:"#7c3aed"}].map((x,M)=>l.jsxs("div",{children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:11,fontFamily:"'DM Mono',monospace",marginBottom:4},children:[l.jsx("span",{style:{color:"rgba(160,180,220,0.7)"},children:x.label}),l.jsxs("span",{style:{color:x.col},children:[(x.val*100).toFixed(1),"% +/- ",(x.std*100).toFixed(1),"%"]})]}),l.jsxs("div",{style:{position:"relative",height:8,borderRadius:4,background:"rgba(91,141,239,0.08)"},children:[l.jsx("div",{style:{position:"absolute",height:"100%",left:`${Math.max(0,(x.val-x.std)*100)}%`,width:`${Math.min(100,x.std*200)}%`,background:`${x.col}20`,borderRadius:4}}),l.jsx("div",{style:{height:"100%",width:`${x.val*100}%`,background:`linear-gradient(90deg,${x.col},${x.col}90)`,borderRadius:4,boxShadow:`0 0 8px ${x.col}40`}})]})]},M))})]})]})}function J3({onAnalyze:t}){var Be,de,me;const e=Z.useContext(xa),[n,i]=Z.useState([]),[r,s]=Z.useState(0),[o,a]=Z.useState(0),[c,u]=Z.useState(1),[p,h]=Z.useState(null),[f,g]=Z.useState(1),[x,M]=Z.useState(0),[v,d]=Z.useState(!1),[m,_]=Z.useState(null),[b,w]=Z.useState(""),[A,E]=Z.useState("all"),[y,C]=Z.useState("snr"),[P,I]=Z.useState("desc"),[F,B]=Z.useState(""),[W,V]=Z.useState(""),[G,z]=Z.useState(""),[j,$]=Z.useState(""),[Q,se]=Z.useState("all"),[ae,Ae]=Z.useState(!1),[De,Oe]=Z.useState([]),[D,q]=Z.useState(!1),[ne,oe]=Z.useState(-1),ye=Z.useRef(null),[Le,ht]=Z.useState("browse"),[ve,Ve]=Z.useState(null),[ie,le]=Z.useState(""),[ze,L]=Z.useState(!1),[Re,Ge]=Z.useState(null),[$e,Te]=Z.useState(null),[R,S]=Z.useState(!1),k=Z.useRef(null);Z.useEffect(()=>{let N=!1;return(async()=>{d(!0),_(null);try{const ue=new URLSearchParams({page:f,limit:20,sort_by:y,sort_dir:P,label:A,...Q!=="all"&&{mission:Q},...b&&{search:b},...F&&{min_snr:F},...W&&{max_snr:W},...G&&{min_period:G},...j&&{max_period:j}}),je=await ln(`${jt}/api/catalog/stars?${ue}`),U=await je.json();if(!je.ok)throw new Error(U.error||"Erreur serveur");N||(i(U.stars),s(U.total),a(U.n_planets_filtered??0),u(U.pages),U.stats&&h(U.stats))}catch(ue){N||_(ue.message)}N||d(!1)})(),()=>{N=!0}},[f,x]);const te=()=>{f===1?M(N=>N+1):g(1)},re=async()=>{if(ve){L(!0),Te(null),Ge(null);try{const N=new FormData;N.append("file",ve),ie.trim()&&N.append("target_id",ie.trim());const ce=await ln(`${jt}/api/catalog/upload`,{method:"POST",body:N}),ue=await ce.json();if(!ce.ok)throw new Error(ue.error||"Erreur serveur");Ge(ue)}catch(N){Te(N.message)}L(!1)}},J=N=>{N.preventDefault(),S(!1);const ce=N.dataTransfer.files[0];ce&&Ve(ce)},be=N=>N===1?"#34d399":"#f87171",ge=N=>N>=10?"#34d399":N>=5?"#fbbf24":"#f87171",Ue=[{value:"snr",label:"SNR"},{value:"period",label:"Période"},{value:"depth",label:"Profondeur"},{value:"score",label:"Score BLS"}];return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:16,animation:"fadeIn .5s ease-out"},children:[l.jsx("div",{style:{display:"flex",gap:4},children:[{id:"browse",label:e?"🔭 Nos étoiles":"Parcourir le catalogue"},{id:"upload",label:e?"📂 Ma propre étoile":"Analyser mon CSV"}].map(N=>l.jsx("button",{onClick:()=>ht(N.id),style:{padding:"7px 16px",borderRadius:8,fontSize:11,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:Le===N.id?"rgba(91,141,239,0.15)":"rgba(15,18,30,0.5)",border:`1px solid ${Le===N.id?"rgba(91,141,239,0.35)":"rgba(91,141,239,0.08)"}`,color:Le===N.id?"#5b8def":"rgba(160,180,220,0.5)"},children:N.label},N.id))}),Le==="browse"&&l.jsxs(l.Fragment,{children:[p&&l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(140px,1fr))",gap:10},children:[{emoji:"⭐",val:p.total_stars.toLocaleString(),label:e?"étoiles en stock":"étoiles indexées"},{emoji:"🌍",val:p.n_planets.toLocaleString(),label:e?"planètes probables":"label planète (1)"},{emoji:"❌",val:p.n_non_planets.toLocaleString(),label:e?"non planètes":"label non-planète (0)"},{emoji:"📡",val:p.avg_snr,label:e?"SNR moyen":"SNR moyen (BLS)"}].map((N,ce)=>l.jsxs(Qe,{style:{padding:"12px 14px",textAlign:"center"},children:[l.jsx("div",{style:{fontSize:20,marginBottom:4},children:N.emoji}),l.jsx("div",{style:{fontSize:18,fontWeight:700,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:N.val}),l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.4)",marginTop:2,fontFamily:"'DM Mono',monospace"},children:N.label})]},ce))}),l.jsxs("div",{style:{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"},children:[l.jsxs("div",{style:{flex:1,minWidth:180,position:"relative"},ref:ye,children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",borderRadius:9,overflow:"hidden"},children:[l.jsx(Br,{size:12,style:{color:"rgba(91,141,239,0.4)",marginLeft:10,flexShrink:0}}),l.jsx("input",{value:b,onChange:N=>{const ce=N.target.value;if(w(ce),oe(-1),ce.length>=2){const ue=ce.toLowerCase(),je=Jp.filter(pe=>pe.toLowerCase().startsWith(ue)),U=ce.toLowerCase().startsWith("kic")?Uc.filter(pe=>pe.toLowerCase().includes(ue)).slice(0,4):[],xe=[...new Set([...je,...U])].slice(0,7);Oe(xe),q(xe.length>0)}else Oe([]),q(!1)},onKeyDown:N=>{N.key==="Enter"&&(ne>=0&&De[ne]?(w(De[ne]),q(!1),oe(-1)):te()),D&&(N.key==="ArrowDown"?(N.preventDefault(),oe(ce=>Math.min(ce+1,De.length-1))):N.key==="ArrowUp"?(N.preventDefault(),oe(ce=>Math.max(ce-1,-1))):N.key==="Escape"&&(q(!1),oe(-1)))},onBlur:()=>setTimeout(()=>q(!1),150),onFocus:()=>{De.length>0&&q(!0)},placeholder:e?"Rechercher une étoile…":"Kepler-10, KIC 11446…",style:{flex:1,padding:"8px 10px",background:"transparent",border:"none",outline:"none",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontSize:11}}),b&&l.jsx("button",{onClick:()=>{w(""),Oe([]),q(!1)},style:{background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)",padding:"0 8px"},children:l.jsx(Rx,{size:11})})]}),D&&De.length>0&&l.jsx("div",{style:{position:"absolute",top:"calc(100% + 4px)",left:0,right:0,background:"rgba(8,11,22,0.97)",border:"1px solid rgba(91,141,239,0.2)",borderRadius:9,overflow:"hidden",zIndex:200,boxShadow:"0 8px 24px rgba(0,0,0,0.5)"},children:De.map((N,ce)=>l.jsxs("div",{onMouseDown:()=>{w(N),q(!1),oe(-1)},style:{padding:"8px 12px",cursor:"pointer",fontSize:11,fontFamily:"'DM Mono',monospace",background:ne===ce?"rgba(91,141,239,0.12)":"transparent",color:ne===ce?"#5b8def":"rgba(200,215,240,0.75)",display:"flex",alignItems:"center",gap:8,borderBottom:ce<De.length-1?"1px solid rgba(91,141,239,0.06)":"none",transition:"background .1s"},onMouseEnter:()=>oe(ce),onMouseLeave:()=>oe(-1),children:[l.jsx(Br,{size:10,style:{opacity:.4,flexShrink:0}}),N]},N))})]}),l.jsx("div",{style:{display:"flex",gap:4},children:[{val:"all",label:e?"Toutes":"Tout"},{val:"1",label:e?"🌍 Planètes":"Planète"},{val:"0",label:e?"⭐ Étoiles":"Non-planète"}].map(N=>l.jsx("button",{onClick:()=>E(N.val),style:{padding:"6px 10px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:A===N.val?"rgba(91,141,239,0.15)":"rgba(15,18,30,0.5)",border:`1px solid ${A===N.val?"rgba(91,141,239,0.3)":"rgba(91,141,239,0.08)"}`,color:A===N.val?"#5b8def":"rgba(160,180,220,0.45)"},children:N.label},N.val))}),l.jsx("div",{style:{display:"flex",gap:4},children:[{val:"all",label:"Toutes missions"},{val:"Kepler",label:"Kepler"},{val:"TESS",label:"TESS"}].map(N=>l.jsx("button",{onClick:()=>{se(N.val),g(1),M(ce=>ce+1)},style:{padding:"6px 10px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:Q===N.val?N.val==="TESS"?"rgba(34,211,238,0.12)":N.val==="Kepler"?"rgba(167,139,250,0.12)":"rgba(91,141,239,0.15)":"rgba(15,18,30,0.5)",border:`1px solid ${Q===N.val?N.val==="TESS"?"rgba(34,211,238,0.35)":N.val==="Kepler"?"rgba(167,139,250,0.35)":"rgba(91,141,239,0.3)":"rgba(91,141,239,0.08)"}`,color:Q===N.val?N.val==="TESS"?"#7c3aed":N.val==="Kepler"?"#a78bfa":"#5b8def":"rgba(160,180,220,0.45)"},children:N.label},N.val))}),l.jsx("select",{value:y,onChange:N=>C(N.target.value),style:{padding:"6px 10px",borderRadius:7,fontSize:10,background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:"rgba(160,180,220,0.7)",fontFamily:"'DM Mono',monospace",cursor:"pointer"},children:Ue.map(N=>l.jsx("option",{value:N.value,children:e?`Trier par ${N.label}`:`↕ ${N.label}`},N.value))}),l.jsx("button",{onClick:()=>I(N=>N==="desc"?"asc":"desc"),title:"Inverser l'ordre",style:{padding:"6px 9px",borderRadius:7,background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:"rgba(160,180,220,0.5)",cursor:"pointer",fontSize:12},children:P==="desc"?"↓":"↑"}),l.jsxs("button",{onClick:()=>Ae(N=>!N),style:{padding:"6px 10px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:ae?"rgba(91,141,239,0.12)":"rgba(15,18,30,0.5)",border:`1px solid ${ae?"rgba(91,141,239,0.3)":"rgba(91,141,239,0.08)"}`,color:ae?"#5b8def":"rgba(160,180,220,0.45)",display:"flex",alignItems:"center",gap:5},children:[l.jsx(o3,{size:10})," ",e?"Filtres avancés":"Filtres"]}),l.jsxs("button",{onClick:te,style:{padding:"6px 14px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:"linear-gradient(135deg,rgba(91,141,239,0.18),rgba(124,58,237,0.18))",border:"1px solid rgba(91,141,239,0.25)",color:"#5b8def",display:"flex",alignItems:"center",gap:4},children:[l.jsx(Br,{size:10})," Appliquer"]})]}),ae&&l.jsx(Qe,{style:{padding:"12px 16px"},children:l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(160px,1fr))",gap:10},children:[[{label:e?"SNR minimum":"SNR min",val:F,set:B,placeholder:"ex: 5"},{label:e?"SNR maximum":"SNR max",val:W,set:V,placeholder:"ex: 20"},{label:e?"Période min (jours)":"Période min (j)",val:G,set:z,placeholder:"ex: 1"},{label:e?"Période max (jours)":"Période max (j)",val:j,set:$,placeholder:"ex: 100"}].map((N,ce)=>l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:5,fontFamily:"'DM Mono',monospace"},children:N.label}),l.jsx("input",{type:"number",value:N.val,onChange:ue=>N.set(ue.target.value),placeholder:N.placeholder,style:{width:"100%",padding:"6px 10px",borderRadius:7,fontSize:11,background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",outline:"none"}})]},ce)),l.jsx("div",{style:{display:"flex",alignItems:"flex-end"},children:l.jsx("button",{onClick:()=>{B(""),V(""),z(""),$("")},style:{padding:"6px 10px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:"none",border:"1px solid rgba(248,113,113,0.2)",color:"rgba(248,113,113,0.5)"},children:"Réinitialiser"})})]})}),m&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"8px 12px",borderRadius:8,background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"},children:[l.jsx(hr,{size:12}),m]}),!v&&r>0&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:12,fontSize:10,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace"},children:[l.jsxs("span",{children:[r.toLocaleString()," étoile",r>1?"s":""," · page ",f,"/",c]}),l.jsxs("span",{style:{color:"#34d399"},children:["🌍 ",o," planète",o>1?"s":""]}),l.jsxs("span",{style:{color:"#f87171"},children:["⭐ ",(r-o).toLocaleString()," non-planète",r-o>1?"s":""]})]}),l.jsx(Qe,{style:{padding:0,overflow:"hidden",border:"1px solid rgba(91,141,239,0.1)"},children:l.jsx("div",{style:{overflowX:"auto"},children:l.jsxs("table",{style:{width:"100%",borderCollapse:"separate",borderSpacing:0,fontFamily:"'DM Mono',monospace",fontSize:11},children:[l.jsx("thead",{children:l.jsx("tr",{style:{background:"rgba(91,141,239,0.04)"},children:(e?[{l:"ID",k:null},{l:"Type",k:null},{l:"Periode",k:"period"},{l:"SNR",k:"snr"},{l:"",k:null}]:[{l:"ID",k:null},{l:"Label",k:null},{l:"Periode (j)",k:"period"},{l:"SNR",k:"snr"},{l:"Profondeur (ppm)",k:"depth"},{l:"Score BLS",k:"score"},{l:"Points",k:null},{l:"",k:null}]).map(N=>l.jsxs("th",{onClick:N.k?()=>{y===N.k?I(ce=>ce==="desc"?"asc":"desc"):(C(N.k),I("desc")),M(ce=>ce+1)}:void 0,style:{padding:"10px 12px",textAlign:"left",fontSize:9,color:y===N.k?"#5b8def":"rgba(160,180,220,0.5)",textTransform:"uppercase",letterSpacing:1.2,fontWeight:600,whiteSpace:"nowrap",cursor:N.k?"pointer":"default",userSelect:"none",borderBottom:"1px solid rgba(91,141,239,0.1)",transition:"color .15s"},children:[N.l,N.k&&l.jsx("span",{style:{marginLeft:4,opacity:y===N.k?1:.3},children:y===N.k?P==="desc"?"↓":"↑":"↕"})]},N.l))})}),l.jsx("tbody",{children:v?l.jsx("tr",{children:l.jsxs("td",{colSpan:e?5:8,style:{padding:40,textAlign:"center",color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace"},children:[l.jsx(mi,{size:16,style:{animation:"spin 1s linear infinite",display:"inline-block",marginRight:8}}),"Chargement..."]})}):n.length===0?l.jsx("tr",{children:l.jsxs("td",{colSpan:e?5:8,style:{padding:40,textAlign:"center",color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",fontSize:12},children:[l.jsx(ur,{size:20,style:{opacity:.3,display:"inline-block",marginRight:8,verticalAlign:"middle"}}),"Aucun resultat"]})}):n.map((N,ce)=>{var pe;const ue=ce%2===0,je=N.mission==="TESS"?"#e879a8":"#5b8def",U=N.mission==="TESS"?"rgba(232,121,168,0.1)":"rgba(91,141,239,0.1)",xe=N.mission==="TESS"?"rgba(232,121,168,0.2)":"rgba(91,141,239,0.2)";return l.jsxs("tr",{style:{background:ue?"rgba(91,141,239,0.02)":"transparent",borderBottom:"1px solid rgba(91,141,239,0.06)",transition:"background .15s"},onMouseEnter:Ee=>Ee.currentTarget.style.background="rgba(91,141,239,0.07)",onMouseLeave:Ee=>Ee.currentTarget.style.background=ue?"rgba(91,141,239,0.02)":"transparent",children:[l.jsx("td",{style:{padding:"10px 12px",color:"#e4e8f7",fontWeight:500,borderBottom:"1px solid rgba(91,141,239,0.05)"},children:l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8},children:[l.jsx("div",{style:{width:5,height:5,borderRadius:"50%",flexShrink:0,background:N.label===1?"#34d399":"rgba(160,180,220,0.2)",boxShadow:N.label===1?"0 0 6px rgba(52,211,153,0.4)":"none"}}),l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:3},children:[l.jsx("span",{children:N.name||N.target_id||`KIC ${N.kepid}`}),N.mission&&l.jsx("span",{style:{fontSize:8,padding:"1px 6px",borderRadius:3,fontWeight:600,background:U,color:je,border:`1px solid ${xe}`,textTransform:"uppercase",letterSpacing:.8,width:"fit-content"},children:N.mission})]})]})}),l.jsx("td",{style:{padding:"10px 12px",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:e?l.jsx("span",{style:{fontSize:14},children:N.label===1?"🌍":"⭐"}):l.jsx("span",{style:{padding:"3px 8px",borderRadius:5,fontSize:9,fontWeight:600,background:N.label===1?"rgba(52,211,153,0.1)":"rgba(248,113,113,0.08)",border:`1px solid ${N.label===1?"rgba(52,211,153,0.25)":"rgba(248,113,113,0.2)"}`,color:be(N.label)},children:N.label===1?"Planete":"Non-planete"})}),l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.6)",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:N.period!=null?l.jsxs(l.Fragment,{children:[l.jsx("span",{style:{color:"#e4e8f7",fontWeight:500},children:N.period})," ",l.jsx("span",{style:{fontSize:9,opacity:.5},children:"j"})]}):l.jsx("span",{style:{opacity:.3},children:"—"})}),l.jsx("td",{style:{padding:"10px 12px",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:N.bls_snr!=null?l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:6},children:[l.jsx("div",{style:{width:40,height:4,borderRadius:2,background:"rgba(91,141,239,0.08)",overflow:"hidden"},children:l.jsx("div",{style:{width:`${Math.min(100,N.bls_snr/20*100)}%`,height:"100%",borderRadius:2,background:ge(N.bls_snr),boxShadow:`0 0 4px ${ge(N.bls_snr)}40`}})}),l.jsx("span",{style:{color:ge(N.bls_snr),fontWeight:600,fontSize:11},children:N.bls_snr.toFixed(1)})]}):l.jsx("span",{style:{opacity:.3},children:"—"})}),!e&&l.jsxs(l.Fragment,{children:[l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.5)",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:N.bls_depth_ppm!=null?N.bls_depth_ppm.toLocaleString():l.jsx("span",{style:{opacity:.3},children:"—"})}),l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.5)",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:N.bls_score!=null?N.bls_score.toFixed(3):l.jsx("span",{style:{opacity:.3},children:"—"})}),l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.4)",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:((pe=N.n_points)==null?void 0:pe.toLocaleString())||l.jsx("span",{style:{opacity:.3},children:"—"})})]}),l.jsx("td",{style:{padding:"10px 12px",borderBottom:"1px solid rgba(91,141,239,0.05)"},children:l.jsxs("button",{onClick:()=>t(N.target_id||`KIC ${N.kepid}`),style:{padding:"5px 12px",borderRadius:7,fontSize:9,cursor:"pointer",fontFamily:"'DM Mono',monospace",fontWeight:600,background:"linear-gradient(135deg,rgba(91,141,239,0.12),rgba(124,58,237,0.12))",border:"1px solid rgba(91,141,239,0.25)",color:"#5b8def",whiteSpace:"nowrap",display:"flex",alignItems:"center",gap:5,transition:"all 0.2s",boxShadow:"0 1px 4px rgba(91,141,239,0.08)"},children:[l.jsx(ur,{size:9}),"Analyser"]})})]},N.target_id||N.kepid)})})]})})}),c>1&&l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",gap:6},children:[l.jsx("button",{onClick:()=>g(N=>Math.max(1,N-1)),disabled:f===1,style:{padding:"5px 12px",borderRadius:7,fontSize:10,cursor:f===1?"not-allowed":"pointer",fontFamily:"'DM Mono',monospace",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:f===1?"rgba(91,141,239,0.2)":"#5b8def",opacity:f===1?.5:1},children:"← Préc."}),Array.from({length:Math.min(7,c)},(N,ce)=>{let ue;return c<=7||f<=4?ue=ce+1:f>=c-3?ue=c-6+ce:ue=f-3+ce,l.jsx("button",{onClick:()=>g(ue),style:{padding:"5px 9px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:f===ue?"rgba(91,141,239,0.18)":"rgba(15,18,30,0.6)",border:`1px solid ${f===ue?"rgba(91,141,239,0.35)":"rgba(91,141,239,0.1)"}`,color:f===ue?"#5b8def":"rgba(160,180,220,0.45)",minWidth:30},children:ue},ue)}),l.jsx("button",{onClick:()=>g(N=>Math.min(c,N+1)),disabled:f===c,style:{padding:"5px 12px",borderRadius:7,fontSize:10,cursor:f===c?"not-allowed":"pointer",fontFamily:"'DM Mono',monospace",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:f===c?"rgba(91,141,239,0.2)":"#5b8def",opacity:f===c?.5:1},children:"Suiv. →"})]})]}),Le==="upload"&&l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14},children:[l.jsxs(Qe,{style:{padding:"16px 20px"},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10},children:[l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7"},children:e?"📋 Comment préparer ton fichier ?":"Format CSV requis"}),l.jsxs("button",{onClick:()=>{const N=["time,flux","1.00000,1.000312","1.02034,0.999987","1.04068,1.000105","1.06102,0.999843","1.08136,1.000201","1.10170,0.999765","1.12204,1.000089","1.14238,0.999934","1.16272,1.000267","1.18306,0.999812","1.20340,0.998150","1.22374,0.997823","1.24408,0.998301","1.26442,0.999102","1.28476,1.000198","1.30510,0.999876","1.32544,1.000043","1.34578,0.999921","1.36612,1.000187","1.38646,0.999654"],ce=new Blob([N.join(`
`)],{type:"text/csv"}),ue=URL.createObjectURL(ce),je=document.createElement("a");je.href=ue,je.download="exemple_courbe_lumiere.csv",je.click(),URL.revokeObjectURL(ue)},style:{padding:"5px 12px",borderRadius:6,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",display:"flex",alignItems:"center",gap:5,background:"rgba(91,141,239,0.08)",border:"1px solid rgba(91,141,239,0.25)",color:"#5b8def"},children:["⬇ ",e?"Télécharger un exemple":"Exemple .csv"]})]}),e?l.jsxs("div",{style:{fontSize:12,color:"rgba(200,215,240,0.65)",lineHeight:1.8},children:[l.jsxs("p",{style:{marginBottom:8},children:["Ton fichier doit être un ",l.jsx("strong",{style:{color:"#5b8def"},children:".csv"})," avec au minimum ces deux colonnes :"]}),l.jsxs("div",{style:{fontFamily:"'DM Mono',monospace",fontSize:11,background:"rgba(91,141,239,0.06)",border:"1px solid rgba(91,141,239,0.15)",borderRadius:8,padding:"10px 14px",marginBottom:8},children:["time,flux",l.jsx("br",{}),"1.0,1.0003",l.jsx("br",{}),"1.02,0.9998",l.jsx("br",{}),"1.04,0.9997",l.jsx("br",{}),"..."]}),l.jsxs("ul",{style:{paddingLeft:16,fontSize:11,color:"rgba(160,180,220,0.6)"},children:[l.jsxs("li",{children:[l.jsx("strong",{style:{color:"#e4e8f7"},children:"time"})," — moment de la mesure (en jours, valeur numérique)"]}),l.jsxs("li",{children:[l.jsx("strong",{style:{color:"#e4e8f7"},children:"flux"})," — luminosité de l'étoile (valeur proche de 1.0 normalement)"]}),l.jsxs("li",{children:["Minimum ",l.jsx("strong",{style:{color:"#fbbf24"},children:"50 points"})," de données requis"]}),l.jsx("li",{children:"Les valeurs manquantes (NaN) sont ignorées automatiquement"})]})]}):l.jsxs("div",{style:{fontSize:11,color:"rgba(160,180,220,0.6)",lineHeight:1.7},children:[l.jsxs("div",{style:{fontFamily:"'DM Mono',monospace",background:"rgba(91,141,239,0.06)",border:"1px solid rgba(91,141,239,0.15)",borderRadius:8,padding:"10px 14px",marginBottom:10,fontSize:10},children:[l.jsx("span",{style:{color:"#34d399"},children:"time"}),",",l.jsx("span",{style:{color:"#5b8def"},children:"flux"}),l.jsx("span",{style:{color:"rgba(160,180,220,0.3)"},children:"[,flux_err]"}),l.jsx("br",{}),l.jsx("span",{style:{color:"rgba(160,180,220,0.4)"},children:"1.02345,1.000312"}),l.jsx("br",{}),l.jsx("span",{style:{color:"rgba(160,180,220,0.4)"},children:"1.04321,0.999987"})]}),l.jsxs("ul",{style:{paddingLeft:14,fontSize:10},children:[l.jsxs("li",{children:[l.jsx("strong",{style:{color:"#34d399"},children:"time"})," — temps en jours BKJD ou BJD (numérique)"]}),l.jsxs("li",{children:[l.jsx("strong",{style:{color:"#5b8def"},children:"flux"})," — flux normalisé (proche de 1.0) ou brut"]}),l.jsx("li",{children:"Colonnes supplémentaires ignorées · NaN filtrés automatiquement"}),l.jsx("li",{children:"Minimum 50 points · Encodage UTF-8 · Séparateur virgule"})]})]})]}),l.jsxs("div",{onDragOver:N=>{N.preventDefault(),S(!0)},onDragLeave:()=>S(!1),onDrop:J,onClick:()=>{var N;return(N=k.current)==null?void 0:N.click()},style:{border:`2px dashed ${R?"rgba(91,141,239,0.6)":ve?"rgba(52,211,153,0.4)":"rgba(91,141,239,0.2)"}`,borderRadius:12,padding:"32px 20px",textAlign:"center",cursor:"pointer",background:R?"rgba(91,141,239,0.05)":"rgba(15,18,30,0.4)",transition:"all .2s"},children:[l.jsx("input",{ref:k,type:"file",accept:".csv",style:{display:"none"},onChange:N=>N.target.files[0]&&Ve(N.target.files[0])}),ve?l.jsxs(l.Fragment,{children:[l.jsx("div",{style:{fontSize:24,marginBottom:8},children:"📄"}),l.jsx("div",{style:{fontSize:12,color:"#34d399",fontFamily:"'DM Mono',monospace",fontWeight:600},children:ve.name}),l.jsxs("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",marginTop:4},children:[(ve.size/1024).toFixed(1)," ko · Cliquer pour changer"]})]}):l.jsxs(l.Fragment,{children:[l.jsx("div",{style:{fontSize:28,marginBottom:8},children:"☁️"}),l.jsx("div",{style:{fontSize:12,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace"},children:e?"Glisse ton fichier .csv ici ou clique pour choisir":"Drag & drop .csv · ou cliquer pour parcourir"})]})]}),l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:6,fontFamily:"'DM Mono',monospace"},children:e?"Nom de l'étoile (optionnel)":"Nom / ID cible (optionnel)"}),l.jsx("input",{value:ie,onChange:N=>le(N.target.value),placeholder:e?"ex: Mon étoile préférée":"ex: TIC 123456789",style:{width:"100%",padding:"8px 12px",borderRadius:8,fontSize:11,background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",outline:"none"}})]}),l.jsx("button",{onClick:re,disabled:!ve||ze,style:{padding:"10px 20px",borderRadius:9,fontSize:12,cursor:!ve||ze?"not-allowed":"pointer",fontFamily:"'DM Mono',monospace",opacity:!ve||ze?.6:1,background:"linear-gradient(135deg,rgba(91,141,239,0.2),rgba(124,58,237,0.2))",border:"1px solid rgba(91,141,239,0.3)",color:"#5b8def",display:"flex",alignItems:"center",gap:8,justifyContent:"center"},children:ze?l.jsxs(l.Fragment,{children:[l.jsx(mi,{size:13,style:{animation:"spin 1s linear infinite"}})," Analyse en cours…"]}):l.jsxs(l.Fragment,{children:[l.jsx(Nc,{size:13})," ",e?"Analyser mon étoile !":"Lancer l'analyse"]})}),$e&&l.jsxs("div",{style:{display:"flex",alignItems:"flex-start",gap:8,padding:"10px 14px",borderRadius:8,background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"},children:[l.jsx(hr,{size:13,style:{marginTop:1,flexShrink:0}}),$e]}),Re&&!ze&&l.jsxs(Qe,{glow:!0,style:{padding:16,animation:"fadeIn .4s ease-out"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12},children:[l.jsxs("div",{children:[l.jsx("div",{style:{fontFamily:"'Space Grotesk',sans-serif",fontWeight:600,fontSize:14,color:"#e4e8f7"},children:Re.target}),l.jsxs("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",marginTop:2},children:[(Be=Re.n_points)==null?void 0:Be.toLocaleString()," points · P = ",Re.period_days," j"]})]}),l.jsx("span",{style:{padding:"4px 12px",borderRadius:12,fontSize:11,fontFamily:"'DM Mono',monospace",color:Re.score>=.7?"#34d399":Re.score>=.35?"#fbbf24":"#f87171",background:`${Re.score>=.7?"#34d399":Re.score>=.35?"#fbbf24":"#f87171"}15`,border:`1px solid ${Re.score>=.7?"#34d399":Re.score>=.35?"#fbbf24":"#f87171"}30`},children:Re.verdict})]}),e?l.jsxs("div",{style:{textAlign:"center",padding:"12px 0"},children:[l.jsx("div",{style:{fontSize:40,marginBottom:8},children:Re.score>=.7?"🌍":Re.score>=.35?"🤔":"❌"}),l.jsx("div",{style:{fontSize:15,fontWeight:600,color:Re.score>=.7?"#34d399":Re.score>=.35?"#fbbf24":"#f87171"},children:Re.verdict}),l.jsxs("div",{style:{fontSize:12,color:"rgba(160,180,220,0.5)",marginTop:6},children:["Notre IA est sûre à ",Math.round(Re.score*100),"%"]})]}):l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",gap:16},children:[l.jsx(eh,{score:Re.score,size:110}),l.jsxs("div",{style:{flex:1,fontSize:11,fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.55)"},children:[l.jsxs("div",{children:["Score : ",l.jsxs("span",{style:{color:"#5b8def",fontWeight:600},children:[(Re.score*100).toFixed(1),"%"]})]}),l.jsxs("div",{style:{marginTop:4},children:["Période : ",l.jsxs("span",{style:{color:"#e4e8f7"},children:[Re.period_days," j"]})]}),l.jsxs("div",{style:{marginTop:4},children:["Points analysés : ",l.jsx("span",{style:{color:"#e4e8f7"},children:(de=Re.n_points)==null?void 0:de.toLocaleString()})]})]})]}),((me=Re.data)==null?void 0:me.length)>0&&l.jsx("div",{style:{height:220,borderRadius:8,overflow:"hidden",marginTop:12},children:l.jsx(dc,{data:Re.data,score:Re.score,isLoading:!1})})]})]})]})}function Q3({onLogin:t}){const[e,n]=Z.useState("login"),[i,r]=Z.useState(""),[s,o]=Z.useState(""),[a,c]=Z.useState(!1),[u,p]=Z.useState(null),[h,f]=Z.useState(null),[g,x]=Z.useState(!1),M=async v=>{if(v.preventDefault(),!(!i.trim()||!s)){x(!0),p(null),f(null);try{const m=await fetch(`${jt}${e==="login"?"/api/auth/login":"/api/auth/register"}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:i.trim().toLowerCase(),password:s})}),_=await m.json();if(!m.ok)throw new Error(_.error||"Erreur");if(e==="login")t(_);else{const b=await fetch(`${jt}/api/auth/login`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:i.trim().toLowerCase(),password:s})}),w=await b.json();if(!b.ok)throw new Error(w.error||"Erreur de connexion post-inscription");t(w)}}catch(d){p(d.message)}x(!1)}};return l.jsxs("div",{style:{minHeight:"100vh",background:"linear-gradient(165deg,#030510 0%,#060a14 30%,#0c1222 60%,#0d1030 100%)",display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"'DM Mono',monospace",position:"relative",overflow:"hidden"},children:[l.jsx("style",{children:Px}),l.jsx(Dx,{}),l.jsx("div",{style:{position:"absolute",top:"-20%",left:"-10%",width:"50%",height:"60%",background:"radial-gradient(circle,rgba(91,141,239,0.08) 0%,transparent 70%)",pointerEvents:"none",filter:"blur(60px)"}}),l.jsx("div",{style:{position:"absolute",bottom:"-15%",right:"-10%",width:"45%",height:"55%",background:"radial-gradient(circle,rgba(124,58,237,0.06) 0%,transparent 70%)",pointerEvents:"none",filter:"blur(60px)"}}),l.jsx("div",{style:{position:"absolute",top:"30%",right:"20%",width:"30%",height:"30%",background:"radial-gradient(circle,rgba(232,121,168,0.04) 0%,transparent 70%)",pointerEvents:"none",filter:"blur(40px)"}}),l.jsxs("div",{style:{position:"relative",zIndex:10,width:"100%",maxWidth:420,padding:"0 24px"},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"center",alignItems:"center",marginBottom:24,position:"relative",height:72},children:[l.jsx("div",{style:{width:52,height:52,borderRadius:"50%",background:"radial-gradient(circle at 40% 40%,#fff8e1,#f0c040,#e8a020)",boxShadow:"0 0 30px rgba(240,192,64,0.35),0 0 60px rgba(240,192,64,0.15),0 0 90px rgba(240,192,64,0.08)",position:"relative",animation:"stellar-glow 3s ease-in-out infinite"},children:l.jsx("div",{style:{width:13,height:13,borderRadius:"50%",background:"radial-gradient(circle at 60% 40%,#1a2540,#0c1222)",boxShadow:"inset 1px 1px 3px rgba(255,255,255,0.08),0 0 4px rgba(0,0,0,0.5)",position:"absolute",top:"50%",left:"50%",animation:"transit-orbit 4.5s ease-in-out infinite"}})}),l.jsx("div",{style:{position:"absolute",width:120,height:40,border:"1px solid rgba(91,141,239,0.08)",borderRadius:"50%",top:"50%",left:"50%",transform:"translate(-50%,-50%) rotateX(70deg)",pointerEvents:"none"}})]}),l.jsxs("div",{style:{textAlign:"center",marginBottom:28},children:[l.jsx("h1",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:28,fontWeight:700,background:"linear-gradient(135deg,#5b8def,#7c3aed,#e879a8)",backgroundSize:"200% 200%",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",marginBottom:6,animation:"gradient-shift 6s ease infinite"},children:"ExoPlanet AI"}),l.jsx("p",{style:{fontSize:12,color:"rgba(160,180,220,0.5)",marginBottom:4},children:"Rejoignez la chasse aux exoplanetes"}),l.jsxs("div",{style:{display:"flex",justifyContent:"center",gap:8,marginTop:8},children:[l.jsx("span",{style:{fontSize:9,padding:"3px 8px",borderRadius:4,background:"rgba(91,141,239,0.1)",color:"#5b8def",border:"1px solid rgba(91,141,239,0.2)",fontFamily:"'DM Mono',monospace",letterSpacing:1,textTransform:"uppercase"},children:"Kepler"}),l.jsx("span",{style:{fontSize:9,padding:"3px 8px",borderRadius:4,background:"rgba(232,121,168,0.1)",color:"#e879a8",border:"1px solid rgba(232,121,168,0.2)",fontFamily:"'DM Mono',monospace",letterSpacing:1,textTransform:"uppercase"},children:"TESS"}),l.jsx("span",{style:{fontSize:9,padding:"3px 8px",borderRadius:4,background:"rgba(52,211,153,0.1)",color:"#34d399",border:"1px solid rgba(52,211,153,0.2)",fontFamily:"'DM Mono',monospace",letterSpacing:1,textTransform:"uppercase"},children:"NASA MAST"})]})]}),l.jsxs(Qe,{style:{padding:28,background:"rgba(11,17,32,0.8)",border:"1px solid rgba(91,141,239,0.12)",boxShadow:"0 20px 60px rgba(0,0,0,0.4),0 0 80px rgba(91,141,239,0.05)"},children:[l.jsx("div",{style:{display:"flex",borderRadius:10,overflow:"hidden",border:"1px solid rgba(91,141,239,0.12)",marginBottom:22,background:"rgba(6,10,20,0.5)"},children:[["login","Connexion",y0],["register","Inscription",S0]].map(([v,d,m])=>l.jsxs("button",{onClick:()=>{n(v),p(null),f(null)},style:{flex:1,padding:"9px 0",cursor:"pointer",fontSize:11,fontFamily:"'DM Mono',monospace",border:"none",transition:"all 0.2s",background:e===v?"rgba(91,141,239,0.15)":"transparent",color:e===v?"#5b8def":"rgba(160,180,220,0.35)",display:"flex",alignItems:"center",justifyContent:"center",gap:5},children:[l.jsx(m,{size:12}),d]},v))}),u&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"9px 12px",borderRadius:8,background:"rgba(248,113,113,0.08)",border:"1px solid rgba(248,113,113,0.15)",fontSize:12,color:"#f87171",marginBottom:16},children:[l.jsx(hr,{size:13}),u]}),h&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"9px 12px",borderRadius:8,background:"rgba(52,211,153,0.08)",border:"1px solid rgba(52,211,153,0.15)",fontSize:12,color:"#34d399",marginBottom:16},children:[l.jsx(ma,{size:13}),h]}),l.jsxs("form",{onSubmit:M,children:[[["Identifiant",i,r,"text","simon",Ys],["Mot de passe",s,o,a?"text":"password","••••••••",wx]].map(([v,d,m,_,b,w],A)=>l.jsxs("div",{style:{marginBottom:16},children:[l.jsx("label",{style:{display:"block",fontSize:10,color:"rgba(160,180,220,0.5)",marginBottom:5,textTransform:"uppercase",letterSpacing:1.5},children:v}),l.jsxs("div",{style:{display:"flex",alignItems:"center",background:"rgba(6,10,20,0.6)",border:"1px solid rgba(91,141,239,0.12)",borderRadius:10,overflow:"hidden",transition:"border-color 0.2s"},children:[l.jsx(w,{size:13,style:{color:"rgba(91,141,239,0.4)",marginLeft:12,flexShrink:0}}),l.jsx("input",{value:d,onChange:E=>m(E.target.value),type:_,placeholder:b,style:{flex:1,padding:"11px 10px",background:"transparent",border:"none",outline:"none",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontSize:13}}),A===1&&l.jsx("button",{type:"button",onClick:()=>c(!a),style:{background:"none",border:"none",padding:"8px 12px",cursor:"pointer",color:"rgba(91,141,239,0.4)"},children:a?l.jsx(kf,{size:13}):l.jsx(Of,{size:13})})]})]},A)),l.jsxs("button",{type:"submit",disabled:g,style:{width:"100%",padding:"12px 0",borderRadius:10,marginTop:10,background:"linear-gradient(135deg,#5b8def,#7c3aed)",border:"none",color:"#fff",fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:8,boxShadow:"0 4px 20px rgba(91,141,239,0.25)",transition:"opacity 0.2s,transform 0.2s,box-shadow 0.2s"},children:[g?l.jsx(mi,{size:15,style:{animation:"spin 1s linear infinite"}}):e==="login"?l.jsx(y0,{size:15}):l.jsx(S0,{size:15}),e==="login"?"Se connecter":"Creer mon compte"]})]})]}),l.jsxs("div",{style:{textAlign:"center",marginTop:18},children:[l.jsx("p",{style:{fontSize:10,color:"rgba(160,180,220,0.22)"},children:"ECE Paris — ING4 Group 1"}),l.jsx("p",{style:{fontSize:9,color:"rgba(160,180,220,0.15)",marginTop:3},children:"Transit photometrique · XGBoost + TSFRESH · NASA MAST Archive"})]})]})]})}function ew(t){if(!t)return"—";try{return new Date(t).toLocaleString("fr-FR",{day:"2-digit",month:"2-digit",year:"2-digit",hour:"2-digit",minute:"2-digit"})}catch{return t}}function tw({history:t,onClear:e,onDelete:n,onAnalyze:i}){const r=Z.useContext(xa),[s,o]=Z.useState(!1),a=Z.useRef(null),[c,u]=Z.useState(""),[p,h]=Z.useState("all"),[f,g]=Z.useState("desc"),x=async()=>{if(!s){o(!0),a.current=setTimeout(()=>o(!1),3e3);return}clearTimeout(a.current),o(!1),await e()},M=t.filter(m=>{var _;return c&&!((_=m.target)!=null&&_.toLowerCase().includes(c.toLowerCase()))?!1:p==="planet"?m.score>=.7:p==="fp"?m.score<.35:p==="other"?m.score>=.35&&m.score<.7:!0}).sort((m,_)=>{const b=new Date(m.date),w=new Date(_.date);return f==="desc"?w-b:b-w}),v=()=>{const m=["Cible","Score (%)","Verdict","Période (j)","Mission","Date"],_=M.map(A=>[A.target,(A.score*100).toFixed(1),A.verdict,A.period_days??"",A.mission??"",A.date?new Date(A.date).toLocaleString():""]),b=[m,..._].map(A=>A.map(E=>`"${String(E).replace(/"/g,'""')}"`).join(",")).join(`
`),w=document.createElement("a");w.href=URL.createObjectURL(new Blob([b],{type:"text/csv;charset=utf-8;"})),w.download=`historique_exoplanetes_${new Date().toISOString().slice(0,10)}.csv`,w.click()},d=[{id:"all",label:"Tout",count:t.length},{id:"planet",label:"Planètes",count:t.filter(m=>m.score>=.7).length},{id:"fp",label:"Faux positifs",count:t.filter(m=>m.score<.35).length},{id:"other",label:"Candidats",count:t.filter(m=>m.score>=.35&&m.score<.7).length}];return t.length?l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"},children:[l.jsxs("div",{style:{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"},children:[l.jsx("div",{style:{display:"flex",gap:2,background:"rgba(10,14,26,0.6)",borderRadius:9,padding:3,border:"1px solid rgba(91,141,239,0.1)"},children:d.map(m=>l.jsxs("button",{onClick:()=>h(m.id),style:{padding:"5px 12px",borderRadius:7,fontSize:11,cursor:"pointer",border:"none",fontFamily:"'DM Mono',monospace",transition:"all .15s",background:p===m.id?"rgba(91,141,239,0.18)":"none",color:p===m.id?"#5b8def":"rgba(160,180,220,0.4)",fontWeight:p===m.id?600:400},children:[m.label,l.jsx("span",{style:{marginLeft:5,fontSize:9,opacity:.6},children:m.count})]},m.id))}),l.jsxs("div",{style:{position:"relative",flex:1,minWidth:140},children:[l.jsx(Br,{size:12,style:{position:"absolute",left:10,top:"50%",transform:"translateY(-50%)",color:"rgba(160,180,220,0.3)",pointerEvents:"none"}}),l.jsx("input",{value:c,onChange:m=>u(m.target.value),placeholder:"Rechercher une étoile...",style:{width:"100%",padding:"7px 10px 7px 28px",background:"rgba(10,14,26,0.6)",border:"1px solid rgba(91,141,239,0.1)",borderRadius:8,color:"#e4e8f7",outline:"none",fontFamily:"'DM Mono',monospace",fontSize:12,boxSizing:"border-box"}}),c&&l.jsx("button",{onClick:()=>u(""),style:{position:"absolute",right:8,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"},children:l.jsx(Rx,{size:12})})]}),l.jsxs("button",{onClick:()=>g(m=>m==="desc"?"asc":"desc"),style:{padding:"7px 12px",borderRadius:8,fontSize:11,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:"rgba(10,14,26,0.6)",border:"1px solid rgba(91,141,239,0.1)",color:"rgba(160,180,220,0.5)",display:"flex",alignItems:"center",gap:6,whiteSpace:"nowrap",transition:"all .15s"},children:[l.jsx(Uf,{size:12}),f==="desc"?"Plus récent":"Plus ancien"]})]}),l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8},children:[l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.2},children:M.length!==t.length?`${M.length} / ${t.length} résultat${M.length>1?"s":""}`:`${t.length} analyse${t.length>1?"s":""}`}),l.jsxs("div",{style:{display:"flex",gap:8},children:[l.jsxs("button",{onClick:v,style:{padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:"rgba(91,141,239,0.08)",border:"1px solid rgba(91,141,239,0.2)",color:"#5b8def",display:"flex",alignItems:"center",gap:5,transition:"all .2s"},children:[l.jsx(Tx,{size:11})," Exporter CSV"]}),l.jsx("button",{onClick:x,style:{padding:"5px 12px",borderRadius:7,fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer",background:s?"rgba(248,113,113,0.12)":"rgba(248,113,113,0.05)",border:s?"1px solid rgba(248,113,113,0.5)":"1px solid rgba(248,113,113,0.15)",color:s?"#f87171":"rgba(248,113,113,0.5)",transition:"all .2s"},children:s?"⚠ Confirmer":"🗑 Vider"})]})]}),M.length===0?l.jsx("div",{style:{padding:"40px 0",textAlign:"center",color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",fontSize:12},children:"Aucun résultat pour ces filtres."}):l.jsx("div",{style:{overflowX:"auto"},children:l.jsxs("table",{style:{width:"100%",borderCollapse:"collapse",fontFamily:"'DM Mono',monospace",fontSize:12},children:[l.jsx("thead",{children:l.jsx("tr",{style:{borderBottom:"1px solid rgba(91,141,239,0.1)"},children:(r?["Étoile","Résultat","Date",""]:["Cible","Score","Verdict","Période","Date",""]).map(m=>l.jsx("th",{style:{padding:"8px 12px",textAlign:"left",fontSize:10,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,fontWeight:400},children:m},m))})}),l.jsx("tbody",{children:M.map((m,_)=>{const b=m.score>=.7?"#34d399":m.score>=.35?"#fbbf24":"#f87171",w=m.score>=.7?"🌍":m.score>=.35?"🔶":"⭐";return l.jsxs("tr",{style:{borderBottom:"1px solid rgba(91,141,239,0.05)"},children:[l.jsx("td",{style:{padding:"10px 12px",color:"#e4e8f7",fontWeight:500},children:m.target}),r?l.jsx("td",{style:{padding:"10px 12px",fontSize:18},children:w}):l.jsxs(l.Fragment,{children:[l.jsx("td",{style:{padding:"10px 12px"},children:l.jsxs("span",{style:{color:b,fontWeight:600},children:[(m.score*100).toFixed(1),"%"]})}),l.jsx("td",{style:{padding:"10px 12px"},children:l.jsx("span",{style:{padding:"2px 8px",borderRadius:4,fontSize:10,background:`${b}15`,border:`1px solid ${b}30`,color:b},children:m.verdict})}),l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.5)",fontSize:11},children:m.period_days?`${m.period_days} j`:"—"})]}),l.jsx("td",{style:{padding:"10px 12px",color:"rgba(160,180,220,0.4)",fontSize:11},children:ew(m.date)}),l.jsx("td",{style:{padding:"10px 12px"},children:l.jsxs("div",{style:{display:"flex",gap:6,alignItems:"center"},children:[l.jsxs("button",{onClick:()=>i(m.target),style:{padding:"3px 10px",borderRadius:6,fontSize:9,cursor:"pointer",fontFamily:"'DM Mono',monospace",whiteSpace:"nowrap",background:"rgba(91,141,239,0.08)",border:"1px solid rgba(91,141,239,0.2)",color:"#5b8def",display:"flex",alignItems:"center",gap:4},children:[l.jsx(ra,{size:9}),r?"Revoir":"Analyser"]}),l.jsx("button",{onClick:()=>n(_),style:{padding:"3px 7px",borderRadius:6,fontSize:9,cursor:"pointer",background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",color:"rgba(248,113,113,0.6)",display:"flex",alignItems:"center"},children:l.jsx(L3,{size:9})})]})})]},_)})})]})})]}):l.jsxs("div",{style:{padding:60,textAlign:"center",color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",fontSize:12},children:[l.jsx(Uf,{size:32,style:{opacity:.3,display:"block",margin:"0 auto 12px"}}),"Aucune analyse effectuée pour ce compte"]})}const E0=[{id:"acq",icon:Yp,title:"1 · Acquisition de Données",short:"Téléchargement NASA MAST",desc:"Les données fondamentales proviennent directement des archives de la NASA (Kepler ou TESS). Le flux photométrique brut (lumière reçue) est extrait sous forme de séries temporelles appelées courbes de lumière.",details:["Utilisation de l'API Lightkurve pour cibler un KIC (Kepler Input Catalog).","Extraction automatique du flux PDCSAP (Pre-search Data Conditioning SAP) qui corrige déjà les interférences instrumentales et thermiques du télescope spatial."],tech:"NASA MAST API · Lightkurve"},{id:"pre",icon:$p,title:"2 · Prétraitement du Signal",short:"Nettoyage & Flattening",desc:"La courbe de lumière reçue contient des variations stellaires naturelles (rotation, taches) et du bruit. Il est crucial d'aplanir cette courbe pour n'observer que les chutes rapides de flux.",details:["Retrait drastique des valeurs aberrantes (Outliers > 5σ) et des NaNs.","Application d'un filtre Savitzky-Golay (Flattening) qui capture la ligne de base lente de l'étoile et la divise pour centrer le flux sur 1.0.","Binning adaptatif pour réduire la résolution si l'étoile a plus de 20 000 points afin d'éviter la saturation de la RAM."],tech:"NumPy · SciPy · Pandas"},{id:"bls",icon:ga,title:"3 · Détection BLS",short:"Box Least Squares",desc:"Un algorithme astrophysique classique (BLS) balaie la courbe aplatie à la recherche de signaux périodiques creux ressemblant à une boîte (la forme typique d'un transit planétaire).",details:["Scan systématique des périodes candidates (ex: de 0.5 à 50 jours).","Calcul du SNR (Signal-To-Noise Ratio) de la meilleure période trouvée.","Cette phase produit la période orbitale (T0, durée, profondeur) permettant de « replier » mathématiquement la courbe (Phase Folding)."],tech:"Astropy BLS"},{id:"feat",icon:Fc,title:"4 · Feature Engineering",short:"Extraction TSFRESH",desc:"L'intelligence artificielle classique ne comprend pas bien les séries temporelles géantes. Nous convertissons la forme de la courbe en des centaines de variables statistiques intelligentes.",details:["L'algorithme analyse la courbe repliée et calcule près de 800 statistiques structurées : asymétrie, pics d'autocorrélation, transformées de Fourier...","Un test d'hypothèse drastique (Benjamini-Yekutieli) filtre ces variables pour ne garder que la quarantaine de caractéristiques véritablement corrélées au signal cible."],tech:"TSFRESH (Time Series Feature Extraction)"},{id:"ml",icon:Nc,title:"5 · Classification XGBoost",short:"Inférence du Modèle",desc:"Un modèle d'arbres de décision sur-boostés prend la décision finale. Entraîné sur des milliers d'exemples confirmés par la NASA, il évalue les statistiques et tranche avec précision.",details:["Prise en compte des variables TSFRESH complétée de métadonnées de l'étoile elles-mêmes modélisées (Rayon stellaire, Température effective Teq, Distance galactique...).","Évaluation par Gradient Boosting pour obtenir une probabilité continue de 0 à 100%.","Verdict final : Planète probable, Signal candidat, ou Faux positif éclipsant."],tech:"XGBoost Classifier · Scikit-Learn"}],nw=[{term:"Transit",def:"Diminution temporaire du flux lumineux d'une étoile provoquée par le passage d'une planète devant son disque stellaire."},{term:"Phase folding",def:"Repliement d'une courbe temporelle sur sa période. Tous les transits se superposent en une seule grande chute visible."},{term:"SNR",def:"Signal-to-Noise Ratio (Rapport Signal/Bruit). Amplitude du transit divisée par le bruit moyen."},{term:"BLS",def:`Box Least Squares. L'algorithme roi pour trouver un signal en forme de "boîte" caché dans le bruit.`},{term:"XGBoost",def:"eXtreme Gradient Boosting. Intelligence Artificielle générant un consensus à partir de centaines d'arbres de décision."},{term:"PDCSAP",def:"Pre-search Data Conditioning SAP. Flux lumineux brut corrigé intelligemment par les algorithmes du télescope original."},{term:"KOI / KIC",def:"Kepler Object of Interest (ciblé) et Kepler Input Catalog (inventaire de toutes les étoiles suivies)."},{term:"ppm",def:"Parts per million (10^-6). Le passage de Jupiter devant le Soleil provoque une baisse de 10 000 ppm (1%). La Terre : 84 ppm."}];function iw({item:t}){const[e,n]=Z.useState(!1),[i,r]=Z.useState(!1);return l.jsx("div",{style:{perspective:1e3,height:160,cursor:"pointer"},onMouseEnter:()=>r(!0),onMouseLeave:()=>r(!1),onClick:()=>n(!e),children:l.jsxs("div",{style:{position:"relative",width:"100%",height:"100%",transition:"transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",transformStyle:"preserve-3d",transform:e?"rotateX(180deg)":"rotateX(0deg)"},children:[l.jsxs(Qe,{style:{position:"absolute",width:"100%",height:"100%",padding:"18px 20px",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",backfaceVisibility:"hidden",background:i?"rgba(91,141,239,0.08)":"rgba(15,18,30,0.5)",border:i?"1px solid rgba(91,141,239,0.3)":"1px solid rgba(91,141,239,0.06)",transition:"all 0.3s"},children:[l.jsx("div",{style:{fontSize:20,fontWeight:700,color:i?"#7c3aed":"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif",textAlign:"center",transition:"color 0.3s"},children:t.term}),l.jsx("div",{style:{fontSize:12,color:"#5b8def",fontFamily:"'DM Mono',monospace",marginTop:16,opacity:i?1:0,transition:"all 0.3s",transform:i?"translateY(0)":"translateY(10px)"},children:"Qu'est-ce que c'est ? (Cliquez)"})]}),l.jsx(Qe,{style:{position:"absolute",width:"100%",height:"100%",padding:"20px 24px",display:"flex",flexDirection:"column",justifyContent:"center",alignItems:"center",backfaceVisibility:"hidden",transform:"rotateX(180deg)",background:"linear-gradient(135deg, rgba(30,15,40,0.95), rgba(40,20,60,0.95))",border:"1px solid rgba(124,58,237,0.4)",boxShadow:"0 0 20px rgba(124,58,237,0.15)"},children:l.jsx("div",{style:{fontSize:13,color:"rgba(230,230,255,0.9)",lineHeight:1.6,fontFamily:"'DM Mono',monospace",textAlign:"center"},children:t.def})})]})})}function Nx(){const[t,e]=Z.useState(E0[0]);return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:24,animation:"fadeIn .5s ease-out"},children:[l.jsxs(Qe,{glow:!0,style:{padding:"28px 32px",background:"linear-gradient(135deg, rgba(8,11,22,0.8), rgba(20,25,45,0.8))",border:"1px solid rgba(91,141,239,0.15)"},children:[l.jsx("h2",{style:{fontSize:24,fontFamily:"'Space Grotesk',sans-serif",color:"#e4e8f7",marginBottom:12},children:"Comprendre ExoPlanet AI"}),l.jsx("p",{style:{color:"rgba(160,180,220,0.7)",fontSize:13,lineHeight:1.6,maxWidth:850,fontFamily:"'DM Mono',monospace"},children:"Découvrez comment notre architecture convertit les ondes lumineuses brutes captées dans l'espace en prédictions intelligentes. Explorez ci-dessous la séquence logicielle qui permet de dénicher l'empreinte d'autres mondes, étape par étape."})]}),l.jsx("h3",{style:{fontSize:14,color:"#e4e8f7",marginTop:8,fontFamily:"'Space Grotesk',sans-serif",textTransform:"uppercase",letterSpacing:1.5},children:"Architecture du Pipeline"}),l.jsxs("div",{style:{display:"flex",gap:24,flexWrap:"wrap"},children:[l.jsx("div",{style:{flex:"1 1 300px",display:"flex",flexDirection:"column",gap:12},children:E0.map((n,i)=>l.jsxs("button",{onClick:()=>e(n),style:{display:"flex",alignItems:"center",gap:16,padding:"16px 20px",borderRadius:12,border:"none",cursor:"pointer",textAlign:"left",background:t.id===n.id?"rgba(91,141,239,0.12)":"rgba(15,18,30,0.5)",border:`1px solid ${t.id===n.id?"rgba(91,141,239,0.4)":"rgba(91,141,239,0.05)"}`,boxShadow:t.id===n.id?"0 4px 20px rgba(91,141,239,0.15)":"none",transition:"all 0.3s cubic-bezier(0.25, 1, 0.5, 1)",transform:t.id===n.id?"translateX(6px)":"none"},children:[l.jsx("div",{style:{width:42,height:42,borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",background:t.id===n.id?"linear-gradient(135deg,#5b8def,#7c3aed)":"rgba(91,141,239,0.05)",color:t.id===n.id?"#fff":"rgba(160,180,220,0.5)",transition:"all 0.3s"},children:l.jsx(n.icon,{size:20})}),l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:13,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:t.id===n.id?"#fff":"#e4e8f7",transition:"color 0.3s"},children:n.title}),l.jsx("div",{style:{fontSize:11,fontFamily:"'DM Mono',monospace",color:t.id===n.id?"rgba(255,255,255,0.7)":"rgba(160,180,220,0.4)",marginTop:4},children:n.short})]})]},n.id))}),l.jsxs(Qe,{style:{flex:"2 1 500px",minHeight:460,position:"relative",overflow:"hidden",display:"flex",flexDirection:"column",padding:"32px"},children:[l.jsx(t.icon,{size:300,style:{position:"absolute",bottom:-40,right:-40,opacity:.03,color:"#5b8def",transform:"rotate(-15deg)",transition:"all 0.5s ease-out"}}),l.jsxs("div",{style:{animation:"slideIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)",zIndex:1,display:"flex",flexDirection:"column",height:"100%"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:12,marginBottom:24},children:[l.jsx("div",{style:{padding:"6px 14px",borderRadius:6,background:"rgba(91,141,239,0.15)",border:"1px solid rgba(91,141,239,0.3)",color:"#5b8def",fontSize:10,fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.5,fontWeight:600},children:"Détails Techniques"}),l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace"},children:t.tech})]}),l.jsx("h3",{style:{fontSize:26,fontWeight:700,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif",marginBottom:16},children:t.title}),l.jsx("p",{style:{fontSize:13,color:"rgba(160,180,220,0.8)",lineHeight:1.8,fontFamily:"'DM Mono',monospace",marginBottom:32},children:t.desc}),l.jsxs("div",{style:{flex:1},children:[l.jsx("div",{style:{fontSize:11,textTransform:"uppercase",letterSpacing:1.5,color:"rgba(160,180,220,0.4)",marginBottom:16,fontFamily:"'DM Mono',monospace"},children:"Dans le code backend :"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:12},children:t.details.map((n,i)=>l.jsx("div",{style:{padding:"16px 20px",borderRadius:10,background:"rgba(0,0,0,0.25)",borderLeft:"3px solid #7c3aed",fontSize:12,color:"rgba(160,180,220,0.9)",lineHeight:1.7,fontFamily:"'DM Mono',monospace",boxShadow:"inset 0 0 10px rgba(0,0,0,0.2)"},children:n},i))})]})]},t.id)]})]}),l.jsx("h3",{style:{fontSize:14,color:"#e4e8f7",marginTop:20,fontFamily:"'Space Grotesk',sans-serif",textTransform:"uppercase",letterSpacing:1.5},children:"Glossaire & Astrométrie"}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))",gap:14},children:nw.map((n,i)=>l.jsx(iw,{item:n},i))})]})}const rw=["TIC 231670397","TIC 16288184","TIC 144065872","TIC 66818296","TIC 317060587","TIC 366989877","TIC 320004517","TIC 38846515","TIC 149601557","TIC 400595342","TIC 304021498","TIC 363260203","TIC 261136679","TIC 55525572","TIC 192826603"],sw=["TOI-700","TOI-700d","TOI-700e","TOI-1338","TOI-1338b","TOI-1452","TOI-1452b","TOI-849","TOI-849b","TOI-125","TOI-125b","TOI-125c","TOI-178","TOI-178b","TOI-178c","TOI-270","TOI-270b","TOI-270c","TOI-270d","TOI-1266","TOI-1266b","TOI-776","TOI-776b","TOI-776c","TOI-431","TOI-431b","TOI-431d","TOI-561","TOI-561b","TOI-561c","TOI-1231","TOI-1231b","TOI-1259","TOI-1259b","TOI-1634","TOI-1634b","TOI-132","TOI-132b","TOI-469","TOI-469b","TOI-469c","TOI-469d","TOI-530","TOI-530b","TOI-1410","TOI-1410b","TOI-2076","TOI-2076b","TOI-2076c","TOI-2076d","TOI-1233","TOI-1233b","TOI-1233c","TOI-1233d","TOI-1233e","TOI-174","TOI-174b","TOI-396","TOI-396b","TOI-396c","TOI-5205","TOI-5205b","TOI-4010","TOI-4010b","TOI-4010c","TOI-4010d","TOI-5688","TOI-5688b","TOI-1453","TOI-1453b","TOI-421","TOI-421b","TOI-421c","TOI-1685","TOI-1685b","TOI-1728","TOI-1728b","TOI-2068","TOI-2068b","TOI-1444","TOI-1444b","TOI-1749","TOI-1749b","TOI-1749c","TOI-1749d"];function ow({current:t,onPick:e}){const[n,i]=Z.useState("kepler"),r=({id:s,label:o,color:a})=>l.jsx("button",{onClick:()=>i(s),style:{flex:1,padding:"5px 0",borderRadius:6,border:"none",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:9,textTransform:"uppercase",letterSpacing:.8,background:n===s?`${a}18`:"transparent",color:n===s?a:"rgba(160,180,220,0.35)",borderBottom:n===s?`2px solid ${a}`:"2px solid transparent",transition:"all .15s"},children:o});return l.jsxs("div",{style:{position:"sticky",top:16,display:"flex",flexDirection:"column",gap:8,maxHeight:"calc(100vh - 160px)"},children:[l.jsx("div",{style:{fontSize:10,fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.35)",textTransform:"uppercase",letterSpacing:1.5,paddingLeft:2,marginBottom:2},children:"Suggestions"}),l.jsx(Qe,{style:{padding:"6px 8px"},children:l.jsxs("div",{style:{display:"flex",gap:2},children:[l.jsx(r,{id:"kepler",label:"Kepler",color:"#5b8def"}),l.jsx(r,{id:"tess",label:"TESS",color:"#e879a8"}),l.jsx(r,{id:"toi",label:"TOI",color:"#a78bfa"}),l.jsx(r,{id:"kic",label:"KIC",color:"rgba(160,180,220,0.5)"})]})}),n==="kepler"&&l.jsxs(Qe,{style:{padding:"10px 12px"},children:[l.jsx("div",{style:{fontSize:9,color:"rgba(91,141,239,0.5)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:8,fontFamily:"'DM Mono',monospace"},children:"Kepler nommées"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:3},children:B3.map(s=>l.jsx("button",{onClick:()=>e(s.id),style:{textAlign:"left",padding:"5px 8px",borderRadius:6,border:"none",background:t===s.id?"rgba(91,141,239,0.15)":"transparent",color:t===s.id?"#5b8def":"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:11,cursor:"pointer",borderLeft:`2px solid ${t===s.id?"#5b8def":"transparent"}`,transition:"all 0.15s"},children:s.label},s.id))})]}),n==="tess"&&l.jsxs(Qe,{style:{padding:"10px 12px",flex:1,overflowY:"auto",minHeight:0},children:[l.jsx("div",{style:{fontSize:9,color:"rgba(232,121,168,0.6)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:8,fontFamily:"'DM Mono',monospace"},children:"Étoiles TESS (TIC)"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:3},children:rw.map(s=>l.jsx("button",{onClick:()=>e(s),style:{textAlign:"left",padding:"4px 8px",borderRadius:6,border:"none",background:t===s?"rgba(232,121,168,0.12)":"transparent",color:t===s?"#e879a8":"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace",fontSize:10,cursor:"pointer",borderLeft:`2px solid ${t===s?"#e879a8":"transparent"}`,transition:"all 0.15s",whiteSpace:"nowrap"},children:s},s))})]}),n==="toi"&&l.jsxs(Qe,{style:{padding:"10px 12px",flex:1,overflowY:"auto",minHeight:0},children:[l.jsx("div",{style:{fontSize:9,color:"rgba(167,139,250,0.6)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:8,fontFamily:"'DM Mono',monospace"},children:"Catalogue TOI"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:3},children:sw.map(s=>l.jsx("button",{onClick:()=>e(s),style:{textAlign:"left",padding:"4px 8px",borderRadius:6,border:"none",background:t===s?"rgba(167,139,250,0.12)":"transparent",color:t===s?"#a78bfa":"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace",fontSize:10,cursor:"pointer",borderLeft:`2px solid ${t===s?"#a78bfa":"transparent"}`,transition:"all 0.15s",whiteSpace:"nowrap"},children:s},s))})]}),n==="kic"&&l.jsxs(Qe,{style:{padding:"10px 12px",flex:1,overflowY:"auto",minHeight:0},children:[l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.35)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:8,fontFamily:"'DM Mono',monospace"},children:"Catalogue KIC"}),l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:3},children:Uc.map(s=>l.jsx("button",{onClick:()=>e(s),style:{textAlign:"left",padding:"4px 8px",borderRadius:6,border:"none",background:t===s?"rgba(91,141,239,0.15)":"transparent",color:t===s?"#5b8def":"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace",fontSize:10,cursor:"pointer",borderLeft:`2px solid ${t===s?"#5b8def":"transparent"}`,transition:"all 0.15s",whiteSpace:"nowrap"},children:s},s))})]})]})}function aw(){const t=Z.useContext(xa),[e,n]=Z.useState([{id:"slot-0",input:"",data:null,loading:!1,error:null},{id:"slot-1",input:"",data:null,loading:!1,error:null}]),i=(u,p)=>n(h=>h.map(f=>f.id===u?{...f,...p}:f)),r=async u=>{const p=e.find(h=>h.id===u);if(!(!p||!p.input.trim())){i(u,{loading:!0,error:null,data:null});try{const h=await ln(`${jt}/api/analyze?id=${encodeURIComponent(p.input.trim())}`),f=await h.json();if(!h.ok)throw new Error(f.error||"Erreur serveur");i(u,{loading:!1,data:f})}catch(h){i(u,{loading:!1,error:h.message})}}},s=u=>{const p=[...Uc,...j3,...Jp],h=p[Math.floor(Math.random()*p.length)];i(u,{input:h})},o=()=>{if(e.length>=3)return;const u=`slot-${Date.now()}`;n(p=>[...p,{id:u,input:"",data:null,loading:!1,error:null}])},a=u=>{e.length<=2||n(p=>p.filter(h=>h.id!==u))},c=u=>u>=.7?"#34d399":u>=.35?"#fbbf24":"#f87171";return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8},children:[l.jsxs("div",{children:[l.jsx("h2",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:15,fontWeight:600,color:"#e4e8f7",marginBottom:2},children:"Comparaison multi-étoiles"}),l.jsx("p",{style:{fontSize:11,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"},children:"Analysez jusqu'à 3 étoiles côte à côte"})]}),l.jsxs("button",{onClick:o,disabled:e.length>=3,style:{display:"flex",alignItems:"center",gap:6,padding:"7px 14px",borderRadius:9,fontSize:11,fontFamily:"'DM Mono',monospace",cursor:e.length>=3?"not-allowed":"pointer",background:"rgba(91,141,239,0.08)",border:"1px solid rgba(91,141,239,0.2)",color:e.length>=3?"rgba(91,141,239,0.3)":"#5b8def",opacity:e.length>=3?.5:1},children:[l.jsx(Ex,{size:12})," Ajouter une étoile"]})]}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:`repeat(${e.length}, 1fr)`,gap:14,alignItems:"start"},children:e.map(u=>{const p=u.data?c(u.data.score):"#5b8def";return l.jsxs(Qe,{style:{padding:14,display:"flex",flexDirection:"column",gap:12},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",gap:6},children:[l.jsxs("span",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,fontFamily:"'DM Mono',monospace"},children:["Étoile ",e.indexOf(u)+1]}),l.jsx("button",{onClick:()=>a(u.id),disabled:e.length<=2,title:"Supprimer ce slot",style:{background:"none",border:"1px solid rgba(248,113,113,0.2)",borderRadius:5,color:e.length<=2?"rgba(248,113,113,0.2)":"rgba(248,113,113,0.6)",cursor:e.length<=2?"not-allowed":"pointer",padding:"2px 6px",fontSize:10,lineHeight:1},children:"✕"})]}),l.jsxs("div",{style:{display:"flex",gap:6},children:[l.jsxs("div",{style:{flex:1,display:"flex",alignItems:"center",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.15)",borderRadius:8,overflow:"hidden"},children:[l.jsx(Br,{size:11,style:{color:"rgba(91,141,239,0.4)",marginLeft:9,flexShrink:0}}),l.jsx("input",{value:u.input,onChange:h=>i(u.id,{input:h.target.value}),onKeyDown:h=>h.key==="Enter"&&r(u.id),placeholder:"Kepler-10, KIC…",style:{flex:1,padding:"7px 8px",background:"transparent",border:"none",outline:"none",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontSize:11}})]}),l.jsx("button",{onClick:()=>s(u.id),title:"Étoile aléatoire",style:{padding:"7px 9px",borderRadius:8,background:"rgba(91,141,239,0.06)",border:"1px solid rgba(91,141,239,0.15)",color:"#5b8def",cursor:"pointer",fontSize:12},children:l.jsx(t3,{size:13})}),l.jsxs("button",{onClick:()=>r(u.id),disabled:u.loading||!u.input.trim(),style:{padding:"7px 11px",borderRadius:8,fontSize:10,fontFamily:"'DM Mono',monospace",cursor:u.loading||!u.input.trim()?"not-allowed":"pointer",background:"linear-gradient(135deg,rgba(91,141,239,0.18),rgba(124,58,237,0.18))",border:"1px solid rgba(91,141,239,0.25)",color:"#5b8def",display:"flex",alignItems:"center",gap:4,flexShrink:0,opacity:u.loading||!u.input.trim()?.6:1},children:[u.loading?l.jsx(mi,{size:11,style:{animation:"spin 1s linear infinite"}}):l.jsx(ra,{size:11}),"Analyser"]})]}),u.loading&&l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",gap:8,padding:20,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace",fontSize:11},children:[l.jsx(mi,{size:16,style:{color:"#5b8def",animation:"spin 1s linear infinite"}}),"Analyse en cours…"]}),u.error&&!u.loading&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"8px 10px",borderRadius:8,background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"},children:[l.jsx(hr,{size:12}),u.error]}),u.data&&!u.loading&&l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:10,animation:"fadeIn .4s ease-out"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",gap:6,flexWrap:"wrap"},children:[l.jsx("span",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600,color:"#e4e8f7"},children:u.data.target}),l.jsx("span",{style:{padding:"3px 10px",borderRadius:12,fontSize:10,fontFamily:"'DM Mono',monospace",color:p,background:`${p}15`,border:`1px solid ${p}30`},children:u.data.verdict})]}),l.jsx("div",{style:{borderRadius:8,overflow:"hidden",height:160,background:"rgba(7,9,15,0.5)"},children:l.jsx(dc,{data:u.data.data||[],score:u.data.score,isLoading:!1})}),l.jsxs("div",{style:{fontSize:9,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",textAlign:"center",marginTop:-6},children:["P = ",u.data.period_days," j"]}),t?l.jsxs("div",{style:{textAlign:"center",padding:"8px 0"},children:[l.jsx("div",{style:{fontSize:32,marginBottom:4},children:u.data.score>=.7?"🌍":u.data.score>=.35?"🤔":"❌"}),l.jsx("div",{style:{fontSize:12,fontWeight:600,color:p,fontFamily:"'Space Grotesk',sans-serif"},children:u.data.score>=.7?"Planète probable !":u.data.score>=.35?"Pas sûr…":"Probablement pas"}),l.jsxs("div",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",marginTop:3,fontFamily:"'DM Mono',monospace"},children:["IA sûre à ",Math.round(u.data.score*100),"%"]})]}):l.jsx("div",{style:{display:"flex",justifyContent:"center"},children:l.jsx(eh,{score:u.data.score,size:120})}),!t&&u.data.characterization&&l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:9,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,marginBottom:6,fontFamily:"'DM Mono',monospace"},children:"Caractéristiques"}),l.jsx(Lx,{data:u.data})]})]}),!u.data&&!u.loading&&!u.error&&l.jsxs("div",{style:{padding:24,textAlign:"center",color:"rgba(160,180,220,0.2)",fontFamily:"'DM Mono',monospace",fontSize:11},children:[l.jsx(ur,{size:24,style:{opacity:.25,display:"block",margin:"0 auto 8px"}}),"Entrez un identifiant puis cliquez Analyser"]})]},u.id)})})]})}function lw({step:t,onNext:e,onSkip:n}){const i=Dl[t],[r,s]=Z.useState(null),o=12;Z.useEffect(()=>{if(!i.sel){s(null);return}const f=document.querySelector(i.sel);if(f){f.scrollIntoView({behavior:"smooth",block:"nearest"});const g=setTimeout(()=>s(f.getBoundingClientRect()),120);return()=>clearTimeout(g)}s(null)},[t,i.sel]);const a=window.innerWidth,c=!i.sel||!r,u=340;let p=0,h=0;return!c&&r&&(p=r.bottom+o+10,h=Math.max(12,Math.min(r.left,a-u-12)),p+200>window.innerHeight&&(p=r.top-200-o)),l.jsxs("div",{style:{position:"fixed",inset:0,zIndex:9990},children:[c&&l.jsx("div",{style:{position:"fixed",inset:0,background:"rgba(2,4,12,0.85)",pointerEvents:"all"}}),r&&l.jsx("div",{style:{position:"fixed",left:r.left-o,top:r.top-o,width:r.width+o*2,height:r.height+o*2,borderRadius:12,boxShadow:["0 0 0 9999px rgba(2,4,12,0.85)","0 0 0 2px rgba(91,141,239,1)","0 0 32px 8px rgba(91,141,239,0.55)"].join(", "),zIndex:9995,pointerEvents:"none",transition:"all .35s cubic-bezier(.4,0,.2,1)"}}),r&&l.jsx("div",{style:{position:"fixed",inset:0,zIndex:9994,pointerEvents:"all"},onClick:e}),l.jsxs("div",{style:{position:"fixed",...c?{top:"50%",left:"50%",transform:"translate(-50%,-50%)"}:{top:p,left:h},width:u,background:"rgba(8,11,22,0.98)",border:"1px solid rgba(91,141,239,0.35)",borderRadius:14,padding:"20px 22px",zIndex:9999,pointerEvents:"all",boxShadow:"0 12px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(91,141,239,0.1)",animation:"fadeIn .22s ease"},children:[l.jsx("div",{style:{fontSize:15,fontWeight:700,color:"#e4e8f7",marginBottom:10,fontFamily:"'Space Grotesk',sans-serif"},children:i.title}),l.jsx("p",{style:{fontSize:12,color:"rgba(200,215,240,0.72)",lineHeight:1.75,marginBottom:18,fontFamily:"'Space Grotesk',sans-serif"},children:i.desc}),l.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between"},children:[l.jsxs("span",{style:{fontSize:9,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"},children:[t+1," / ",Dl.length]}),l.jsxs("div",{style:{display:"flex",gap:8},children:[l.jsx("button",{onClick:n,style:{padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",background:"none",border:"1px solid rgba(160,180,220,0.15)",color:"rgba(160,180,220,0.4)"},children:"Passer"}),l.jsx("button",{onClick:e,style:{padding:"5px 18px",borderRadius:7,fontSize:10,cursor:"pointer",fontFamily:"'DM Mono',monospace",fontWeight:600,background:"linear-gradient(135deg,rgba(91,141,239,0.22),rgba(124,58,237,0.22))",border:"1px solid rgba(91,141,239,0.4)",color:"#5b8def"},children:t===Dl.length-1?"Terminer ✓":"Suivant →"})]})]})]})]})}const pl=[{id:"compte",label:"Compte",icon:Ys,items:[{id:"identite",label:"Identité"},{id:"session",label:"Session"}]},{id:"securite",label:"Identifiants",icon:wx,items:[{id:"pseudo",label:"Nom d'utilisateur"},{id:"motdepasse",label:"Mot de passe"}]},{id:"apparence",label:"Apparence",icon:x3,items:[{id:"avatar",label:"Avatar"},{id:"affichage",label:"Affichage"}]},{id:"donnees",label:"Données",icon:bx,items:[{id:"stats",label:"Statistiques"},{id:"realisations",label:"Réalisations"},{id:"csv",label:"Imports CSV"}]},{id:"activite",label:"Activité",icon:Uf,items:[{id:"historique",label:"Historique"}]}],Bf=[{id:"rocket",icon:M3},{id:"satellite",icon:T3},{id:"moon",icon:_3},{id:"orbit",icon:ga},{id:"ghost",icon:l3},{id:"user",icon:Ys}];function cw({authState:t,history:e,onLogout:n,setAuthState:i,isLightMode:r,setIsLightMode:s,onAnalyze:o,onClearHistory:a,onDeleteHistory:c}){var Ae,De,Oe;const[u,p]=Z.useState("compte"),[h,f]=Z.useState("identite"),g=e.length,x=e.filter(D=>{var q;return(q=D.verdict)==null?void 0:q.toLowerCase().includes("planète")}).length,M=e.filter(D=>{var q;return(q=D.verdict)==null?void 0:q.toLowerCase().includes("faux positif")}).length,v=e.filter(D=>D.mission==="Custom CSV"),d=g>0?Math.round(x/g*100):0,m=((Ae=Bf.find(D=>D.id===(t==null?void 0:t.avatar)))==null?void 0:Ae.icon)||Ys,_=async D=>{try{(await ln(`${jt}/api/auth/update_profile`,{method:"POST",body:JSON.stringify({avatar:D})})).ok&&i(ne=>({...ne,avatar:D}))}catch(q){console.error(q)}},[b,w]=Z.useState({old:"",new:"",showOld:!1,showNew:!1}),[A,E]=Z.useState(null),y=async D=>{D.preventDefault(),E(null);try{const q=await ln(`${jt}/api/auth/change_password`,{method:"POST",body:JSON.stringify({old_password:b.old,new_password:b.new})}),ne=await q.json();if(!q.ok)throw new Error(ne.error||"Erreur serveur");E({ok:!0,msg:"Mot de passe modifié avec succès."}),w({old:"",new:"",showOld:!1,showNew:!1})}catch(q){E({ok:!1,msg:q.message})}},[C,P]=Z.useState(""),[I,F]=Z.useState(null),B=async D=>{if(D.preventDefault(),F(null),!C||C.length<3)return F({ok:!1,msg:"Pseudo trop court (min. 3 caractères)"});try{const q=await ln(`${jt}/api/auth/change_username`,{method:"POST",body:JSON.stringify({new_username:C})}),ne=await q.json();if(!q.ok)throw new Error(ne.error||"Erreur serveur");F({ok:!0,msg:"Pseudo modifié avec succès !"}),i(oe=>({...oe,username:C,token:ne.token})),P("")}catch(q){F({ok:!1,msg:q.message})}},W=D=>{var q;if(u===D)p(null);else{p(D);const ne=pl.find(oe=>oe.id===D);(q=ne==null?void 0:ne.items)!=null&&q.length&&f(ne.items[0].id)}},V=[{id:"novice",label:"Novice",icon:sa,desc:"Première analyse effectuée",unlocked:g>=1,color:"#5b8def",current:Math.min(g,1),max:1},{id:"explorateur",label:"Explorateur",icon:Zp,desc:"10 analyses complétées",unlocked:g>=10,color:"#a78bfa",current:Math.min(g,10),max:10},{id:"chasseur",label:"Chasseur",icon:Fc,desc:"Première planète détectée",unlocked:x>=1,color:"#34d399",current:Math.min(x,1),max:1},{id:"veteran",label:"Vétéran",icon:Nc,desc:"50 analyses complétées",unlocked:g>=50,color:"#f59e0b",current:Math.min(g,50),max:50},{id:"chercheur",label:"Chercheur",icon:ur,desc:"5 planètes détectées",unlocked:x>=5,color:"#06b6d4",current:Math.min(x,5),max:5},{id:"scientifique",label:"Scientifique",icon:$p,desc:"100 analyses complétées",unlocked:g>=100,color:"#ec4899",current:Math.min(g,100),max:100}],G={width:"100%",padding:"10px 14px",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.2)",borderRadius:8,color:"#e4e8f7",outline:"none",fontFamily:"'DM Mono',monospace",fontSize:13,boxSizing:"border-box"},z={display:"block",fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:6,fontFamily:"'DM Mono',monospace",letterSpacing:.5},j={fontSize:11,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.5,marginBottom:6},$=({status:D})=>D?l.jsx("div",{style:{padding:"10px 14px",marginBottom:12,fontSize:12,borderRadius:8,background:D.ok?"rgba(52,211,153,0.1)":"rgba(248,113,113,0.1)",color:D.ok?"#34d399":"#f87171",border:`1px solid ${D.ok?"rgba(52,211,153,0.3)":"rgba(248,113,113,0.3)"}`,fontFamily:"'DM Mono',monospace"},children:D.msg}):null,Q=()=>{switch(h){case"identite":{const D=g>=100?"Scientifique":g>=50?"Vétéran":g>=10?"Explorateur":g>=1?"Novice":"Recrue";return l.jsxs("div",{children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:24,marginBottom:36,flexWrap:"wrap"},children:[l.jsx("div",{style:{width:88,height:88,borderRadius:"50%",background:"linear-gradient(135deg,#5b8def,#7c3aed)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 12px 32px rgba(91,141,239,0.3)",border:"2px solid rgba(255,255,255,0.08)",flexShrink:0},children:l.jsx(m,{size:44,color:"#fff"})}),l.jsxs("div",{children:[l.jsx("h2",{style:{margin:0,fontFamily:"'Space Grotesk',sans-serif",fontSize:24,fontWeight:700,color:"#e4e8f7"},children:t.username}),l.jsx("div",{style:{fontSize:12,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace",marginTop:4},children:"Explorateur stellaire"}),l.jsx("div",{style:{display:"flex",gap:6,marginTop:12,flexWrap:"wrap"},children:V.filter(q=>q.unlocked).map(q=>{const ne=q.icon;return l.jsxs("span",{style:{display:"inline-flex",alignItems:"center",gap:5,padding:"3px 10px",borderRadius:20,background:`${q.color}18`,border:`1px solid ${q.color}44`,fontSize:10,color:q.color,fontFamily:"'DM Mono',monospace"},children:[l.jsx(ne,{size:10}),q.label]},q.id)})})]})]}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12},children:[{label:"Identifiant",value:t.username,color:"#e4e8f7"},{label:"Rang actuel",value:D,color:"#5b8def"},{label:"Analyses réalisées",value:`${g} cibles`,color:"#e4e8f7"},{label:"Planètes détectées",value:`${x} planète${x!==1?"s":""}`,color:"#34d399"}].map(q=>l.jsxs("div",{style:{padding:"16px 20px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(91,141,239,0.08)"},children:[l.jsx("div",{style:j,children:q.label}),l.jsx("div",{style:{fontFamily:"'DM Mono',monospace",fontSize:14,color:q.color},children:q.value})]},q.label))})]})}case"session":return l.jsxs("div",{children:[l.jsx("p",{style:{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:28},children:"Ta session est actuellement active. En te déconnectant, tu seras redirigé vers la page de connexion et toutes les données non sauvegardées seront perdues."}),l.jsxs("div",{style:{padding:"20px 24px",background:"rgba(248,113,113,0.05)",border:"1px solid rgba(248,113,113,0.12)",borderRadius:12},children:[l.jsx("div",{style:{fontSize:11,color:"rgba(248,113,113,0.6)",fontFamily:"'DM Mono',monospace",marginBottom:12,textTransform:"uppercase",letterSpacing:1},children:"Session active"}),l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",gap:16,flexWrap:"wrap"},children:[l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:14,color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontWeight:600},children:t.username}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.4)",marginTop:4},children:"Connecté maintenant"})]}),l.jsxs("button",{onClick:n,style:{padding:"10px 20px",borderRadius:8,background:"rgba(248,113,113,0.1)",border:"1px solid rgba(248,113,113,0.3)",color:"#f87171",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,display:"flex",alignItems:"center",gap:8,transition:"all .2s"},children:[l.jsx(m3,{size:16})," Déconnexion"]})]})]})]});case"pseudo":return l.jsxs("div",{style:{maxWidth:460},children:[l.jsx("p",{style:{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:20},children:"Ton nom d'utilisateur est visible dans ton profil. Il doit contenir au minimum 3 caractères."}),l.jsxs("div",{style:{padding:"14px 18px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(91,141,239,0.08)",marginBottom:24},children:[l.jsx("div",{style:j,children:"Pseudo actuel"}),l.jsx("div",{style:{fontFamily:"'DM Mono',monospace",fontSize:14,color:"#5b8def"},children:t.username})]}),l.jsx($,{status:I}),l.jsxs("form",{onSubmit:B,style:{display:"flex",flexDirection:"column",gap:14},children:[l.jsxs("div",{children:[l.jsx("label",{style:z,children:"Nouveau pseudo"}),l.jsx("input",{type:"text",value:C,onChange:D=>P(D.target.value),placeholder:"Entrez votre nouveau pseudo...",style:G})]}),l.jsx("button",{type:"submit",style:{padding:"12px 24px",borderRadius:8,background:"linear-gradient(135deg,#5b8def,#7c3aed)",border:"none",color:"#fff",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,alignSelf:"flex-start",transition:"all .2s"},children:"Changer le pseudo"})]})]});case"motdepasse":return l.jsxs("div",{style:{maxWidth:460},children:[l.jsx("p",{style:{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:20},children:"Pour modifier ton mot de passe, saisis ton mot de passe actuel puis le nouveau souhaité."}),l.jsx($,{status:A}),l.jsxs("form",{onSubmit:y,style:{display:"flex",flexDirection:"column",gap:14},children:[l.jsxs("div",{children:[l.jsx("label",{style:z,children:"Mot de passe actuel"}),l.jsxs("div",{style:{position:"relative"},children:[l.jsx("input",{type:b.showOld?"text":"password",value:b.old,onChange:D=>w(q=>({...q,old:D.target.value})),placeholder:"••••••••",style:{...G,paddingRight:44}}),l.jsx("button",{type:"button",onClick:()=>w(D=>({...D,showOld:!D.showOld})),style:{position:"absolute",right:12,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"},children:b.showOld?l.jsx(kf,{size:16}):l.jsx(Of,{size:16})})]})]}),l.jsxs("div",{children:[l.jsx("label",{style:z,children:"Nouveau mot de passe"}),l.jsxs("div",{style:{position:"relative"},children:[l.jsx("input",{type:b.showNew?"text":"password",value:b.new,onChange:D=>w(q=>({...q,new:D.target.value})),placeholder:"••••••••",style:{...G,paddingRight:44}}),l.jsx("button",{type:"button",onClick:()=>w(D=>({...D,showNew:!D.showNew})),style:{position:"absolute",right:12,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"},children:b.showNew?l.jsx(kf,{size:16}):l.jsx(Of,{size:16})})]})]}),l.jsx("button",{type:"submit",style:{padding:"12px 24px",borderRadius:8,background:"linear-gradient(135deg,rgba(91,141,239,0.18),rgba(124,58,237,0.18))",border:"1px solid rgba(91,141,239,0.4)",color:"#5b8def",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,alignSelf:"flex-start",transition:"all .2s"},children:"Modifier le mot de passe"})]})]});case"avatar":return l.jsxs("div",{style:{display:"flex",gap:36,alignItems:"flex-start",flexWrap:"wrap"},children:[l.jsxs("div",{style:{display:"flex",flexDirection:"column",alignItems:"center",gap:10},children:[l.jsx("div",{style:{width:96,height:96,borderRadius:"50%",background:"linear-gradient(135deg,#5b8def,#7c3aed)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 12px 32px rgba(91,141,239,0.3)",border:"2px solid rgba(255,255,255,0.08)"},children:l.jsx(m,{size:50,color:"#fff"})}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace"},children:"Avatar actuel"})]}),l.jsxs("div",{style:{flex:1},children:[l.jsx("div",{style:j,children:"Choisir un avatar"}),l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,maxWidth:300,marginTop:12},children:Bf.map(D=>{const q=D.icon,ne=(t==null?void 0:t.avatar)===D.id;return l.jsxs("button",{onClick:()=>_(D.id),style:{padding:20,borderRadius:12,cursor:"pointer",background:ne?"linear-gradient(135deg,rgba(91,141,239,0.2),rgba(124,58,237,0.2))":"rgba(15,18,30,0.6)",border:ne?"2px solid rgba(91,141,239,0.6)":"2px solid rgba(255,255,255,0.05)",display:"flex",flexDirection:"column",alignItems:"center",gap:8,transition:"all .2s"},children:[l.jsx(q,{size:28,color:ne?"#5b8def":"rgba(160,180,220,0.55)"}),l.jsx("span",{style:{fontSize:10,fontFamily:"'DM Mono',monospace",color:ne?"#5b8def":"rgba(160,180,220,0.35)",textTransform:"capitalize"},children:D.id})]},D.id)})})]})]});case"affichage":return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:16,maxWidth:480},children:[l.jsx("p",{style:{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,margin:0},children:"Personnalise l'affichage de l'interface pour un meilleur confort visuel."}),l.jsx("div",{style:{padding:"18px 22px",background:"rgba(15,18,30,0.6)",borderRadius:12,border:"1px solid rgba(91,141,239,0.08)"},children:l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",gap:16},children:[l.jsxs("div",{children:[l.jsx("div",{style:{fontSize:14,color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontWeight:600},children:"Mode Jour"}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.4)",marginTop:4},children:"Inversion visuelle pour plein soleil"})]}),l.jsx("button",{onClick:()=>s(!r),style:{width:48,height:26,borderRadius:13,cursor:"pointer",border:"none",background:r?"#5b8def":"rgba(255,255,255,0.1)",position:"relative",transition:"background .25s",flexShrink:0,padding:0},children:l.jsx("div",{style:{width:20,height:20,borderRadius:"50%",background:"#fff",position:"absolute",top:3,transition:"left .25s",left:r?24:3,boxShadow:"0 1px 4px rgba(0,0,0,0.35)"}})})]})})]});case"stats":return l.jsxs("div",{children:[l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:16,marginBottom:24},children:[{val:g,label:"Cibles analysées",color:"#5b8def",bg:"rgba(91,141,239,0.08)",border:"rgba(91,141,239,0.15)"},{val:x,label:"Planètes probables",color:"#34d399",bg:"rgba(52,211,153,0.08)",border:"rgba(52,211,153,0.15)"},{val:M,label:"Faux positifs écartés",color:"#f87171",bg:"rgba(248,113,113,0.08)",border:"rgba(248,113,113,0.15)"},{val:`${d}%`,label:"Taux de détection",color:"#f59e0b",bg:"rgba(245,158,11,0.08)",border:"rgba(245,158,11,0.15)"}].map(D=>l.jsxs("div",{style:{background:D.bg,borderRadius:12,padding:24,border:`1px solid ${D.border}`,textAlign:"center"},children:[l.jsx("div",{style:{fontSize:36,fontWeight:700,color:D.color,fontFamily:"'DM Mono',monospace"},children:D.val}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:6},children:D.label})]},D.label))}),g>0&&l.jsxs("div",{style:{padding:"18px 22px",background:"rgba(15,18,30,0.6)",borderRadius:12,border:"1px solid rgba(91,141,239,0.08)"},children:[l.jsx("div",{style:j,children:"Répartition des résultats"}),l.jsxs("div",{style:{display:"flex",height:8,borderRadius:4,overflow:"hidden",marginTop:12,gap:1},children:[l.jsx("div",{style:{flex:x,background:"#34d399",minWidth:x>0?4:0,transition:"flex .5s ease"}}),l.jsx("div",{style:{flex:M,background:"#f87171",minWidth:M>0?4:0,transition:"flex .5s ease"}}),l.jsx("div",{style:{flex:Math.max(0,g-x-M),background:"rgba(160,180,220,0.12)"}})]}),l.jsxs("div",{style:{display:"flex",gap:16,marginTop:10,flexWrap:"wrap"},children:[l.jsxs("span",{style:{fontSize:11,color:"#34d399",fontFamily:"'DM Mono',monospace"},children:["● Planètes ",x]}),l.jsxs("span",{style:{fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"},children:["● Faux positifs ",M]}),l.jsxs("span",{style:{fontSize:11,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace"},children:["● Autres ",Math.max(0,g-x-M)]})]})]})]});case"realisations":return l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:10},children:V.map(D=>{const q=D.icon,ne=D.max>1?Math.round(D.current/D.max*100):D.unlocked?100:0;return l.jsx("div",{style:{padding:"14px 18px",borderRadius:12,background:D.unlocked?"rgba(15,18,30,0.7)":"rgba(10,14,26,0.4)",border:D.unlocked?`1px solid ${D.color}30`:"1px solid rgba(255,255,255,0.04)",opacity:D.unlocked?1:.55,transition:"opacity .2s"},children:l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:14},children:[l.jsx("div",{style:{width:42,height:42,borderRadius:10,flexShrink:0,background:D.unlocked?`${D.color}18`:"rgba(15,18,30,0.5)",border:D.unlocked?`1px solid ${D.color}40`:"1px solid rgba(255,255,255,0.05)",display:"flex",alignItems:"center",justifyContent:"center"},children:l.jsx(q,{size:20,color:D.unlocked?D.color:"rgba(160,180,220,0.25)"})}),l.jsxs("div",{style:{flex:1,minWidth:0},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:3},children:[l.jsx("span",{style:{fontSize:14,fontWeight:600,color:D.unlocked?"#e4e8f7":"rgba(160,180,220,0.35)",fontFamily:"'Space Grotesk',sans-serif"},children:D.label}),l.jsxs("span",{style:{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"},children:[D.current,"/",D.max]})]}),l.jsx("div",{style:{fontSize:11,color:"rgba(160,180,220,0.38)",fontFamily:"'DM Mono',monospace",marginBottom:D.max>1?8:0},children:D.desc}),D.max>1&&l.jsx("div",{style:{height:3,borderRadius:2,background:"rgba(255,255,255,0.05)",overflow:"hidden"},children:l.jsx("div",{style:{width:`${ne}%`,height:"100%",background:D.unlocked?D.color:"rgba(160,180,220,0.15)",borderRadius:2,transition:"width .6s ease"}})})]}),D.unlocked&&l.jsx(ma,{size:18,color:D.color,style:{flexShrink:0}})]})},D.id)})});case"csv":return v.length===0?l.jsx("div",{style:{textAlign:"center",padding:"56px 0",color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",fontSize:13},children:"Aucun import CSV pour le moment."}):l.jsx("div",{style:{display:"flex",flexDirection:"column",gap:8},children:v.map((D,q)=>l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 16px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(255,255,255,0.05)"},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:12},children:[l.jsx(Tx,{size:14,color:"rgba(160,180,220,0.4)"}),l.jsx("span",{style:{fontSize:13,fontFamily:"'DM Mono',monospace",color:"#e4e8f7"},children:D.target})]}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:14},children:[l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.4)"},children:new Date(D.date).toLocaleDateString()}),l.jsx("span",{style:{fontSize:11,padding:"3px 10px",borderRadius:20,background:D.verdict==="Planète probable"?"rgba(52,211,153,0.1)":"rgba(248,113,113,0.1)",color:D.verdict==="Planète probable"?"#34d399":"#f87171",border:D.verdict==="Planète probable"?"1px solid rgba(52,211,153,0.2)":"1px solid rgba(248,113,113,0.2)",fontFamily:"'DM Mono',monospace"},children:D.verdict})]})]},q))});case"historique":return l.jsx(tw,{history:e,onClear:a,onDelete:c,onAnalyze:o});case"documentation":return l.jsx(Nx,{});default:return null}},se=((De=pl.flatMap(D=>D.items).find(D=>D.id===h))==null?void 0:De.label)||"",ae=((Oe=pl.find(D=>D.items.some(q=>q.id===h)))==null?void 0:Oe.label)||"";return l.jsxs("div",{style:{display:"flex",minHeight:600,background:"rgba(5,8,18,0.5)",borderRadius:16,border:"1px solid rgba(91,141,239,0.08)",overflow:"hidden",animation:"fadeIn .5s ease-out"},children:[l.jsxs("div",{style:{width:228,flexShrink:0,borderRight:"1px solid rgba(91,141,239,0.07)",display:"flex",flexDirection:"column",background:"rgba(4,6,15,0.6)"},children:[l.jsxs("div",{style:{padding:"22px 18px",borderBottom:"1px solid rgba(91,141,239,0.07)",display:"flex",alignItems:"center",gap:12},children:[l.jsx("div",{style:{width:42,height:42,borderRadius:"50%",flexShrink:0,background:"linear-gradient(135deg,#5b8def,#7c3aed)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 4px 12px rgba(91,141,239,0.22)"},children:l.jsx(m,{size:22,color:"#fff"})}),l.jsxs("div",{style:{minWidth:0},children:[l.jsx("div",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"},children:t.username}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.38)",fontFamily:"'DM Mono',monospace",marginTop:1},children:"Explorateur stellaire"})]})]}),l.jsx("nav",{style:{flex:1,padding:"6px 0",overflowY:"auto"},children:pl.map(D=>{const q=D.icon,ne=u===D.id;return l.jsxs("div",{children:[l.jsxs("button",{onClick:()=>W(D.id),style:{width:"100%",display:"flex",alignItems:"center",gap:10,padding:"9px 18px",background:"none",border:"none",cursor:"pointer",color:ne?"#e4e8f7":"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace",fontSize:12,fontWeight:ne?600:400,textAlign:"left",transition:"color .15s"},children:[l.jsx(q,{size:14,style:{flexShrink:0}}),l.jsx("span",{style:{flex:1},children:D.label}),l.jsx(ra,{size:12,style:{transform:ne?"rotate(90deg)":"rotate(0deg)",transition:"transform .2s",color:"rgba(160,180,220,0.22)"}})]}),ne&&D.items.map(oe=>l.jsx("button",{onClick:()=>f(oe.id),style:{width:"100%",display:"flex",alignItems:"center",padding:"7px 18px 7px 42px",background:h===oe.id?"rgba(91,141,239,0.09)":"none",border:"none",borderLeft:h===oe.id?"2px solid #5b8def":"2px solid transparent",cursor:"pointer",color:h===oe.id?"#5b8def":"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",fontSize:12,textAlign:"left",transition:"all .15s"},children:oe.label},oe.id))]},D.id)})})]}),l.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",minWidth:0},children:[l.jsxs("div",{style:{padding:"16px 30px",borderBottom:"1px solid rgba(91,141,239,0.07)",display:"flex",alignItems:"center",gap:8},children:[l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"},children:ae}),l.jsx(ra,{size:11,color:"rgba(160,180,220,0.18)"}),l.jsx("span",{style:{fontSize:11,color:"rgba(160,180,220,0.65)",fontFamily:"'DM Mono',monospace",fontWeight:600},children:se})]}),l.jsxs("div",{style:{flex:1,padding:"26px 30px",overflowY:"auto"},children:[l.jsx("h2",{style:{margin:"0 0 4px",fontFamily:"'Space Grotesk',sans-serif",fontSize:20,fontWeight:700,color:"#e4e8f7"},children:se}),l.jsx("div",{style:{height:1,background:"rgba(91,141,239,0.07)",margin:"14px 0 22px"}}),Q()]})]})]})}function uw(){var Ve;const[t,e]=Z.useState(Ix()),[n,i]=Z.useState("analysis"),[r,s]=Z.useState(()=>localStorage.getItem("simpleMode")==="true"),[o,a]=Z.useState(!1),[c,u]=Z.useState(0),[p,h]=Z.useState(!1);Z.useEffect(()=>{localStorage.setItem("simpleMode",r)},[r]),Z.useEffect(()=>{t&&t.has_seen_tutorial===!1&&(a(!0),u(0))},[t]);const f=()=>{a(!1),localStorage.setItem("tourDone","1"),t&&t.has_seen_tutorial===!1&&ln(`${jt}/api/auth/tutorial_seen`,{method:"POST"}).then(()=>e(ie=>({...ie,has_seen_tutorial:!0}))).catch(console.error)},g=()=>{c>=Dl.length-1?f():u(ie=>ie+1)},x=()=>{f()},[M,v]=Z.useState("Kepler-10"),[d,m]=Z.useState("Kepler-10"),[_,b]=Z.useState([]),[w,A]=Z.useState(!1),[E,y]=Z.useState(-1),C=Z.useRef(null),[P,I]=Z.useState(null),[F,B]=Z.useState(!1),[W,V]=Z.useState(null),[G,z]=Z.useState(null),[j,$]=Z.useState({visible:!1,stepIdx:0,pct:0,waiting:!1}),[Q,se]=Z.useState([]),[ae,Ae]=Z.useState(null),De=Z.useRef(null),Oe=Z.useRef(null),D=ie=>{const le={token:ie.token,username:ie.username,has_seen_tutorial:ie.has_seen_tutorial};V3(le),e(le)},q=()=>{Ll(),e(null),I(null),Ae(null)};Z.useEffect(()=>{t&&(ln(`${jt}/api/status`).then(ie=>ie.json()).then(Ae).catch(()=>{Ll(),e(null)}),ln(`${jt}/api/history`).then(ie=>ie.json()).then(ie=>{Array.isArray(ie)&&se(ie)}).catch(()=>{}))},[t]);const ne=()=>{$({visible:!0,stepIdx:0,pct:zf[0].pct,waiting:!1});const ie=[0,600,1200,1900,2600];Oe.current=[],ie.forEach((le,ze)=>{Oe.current.push(setTimeout(()=>{$({visible:!0,stepIdx:ze,pct:zf[ze].pct,waiting:!1})},le))}),Oe.current.push(setTimeout(()=>{$(le=>le.pct<100?{...le,waiting:!0}:le)},3400))},oe=()=>{(Oe.current||[]).forEach(clearTimeout),$({visible:!0,stepIdx:5,pct:100,waiting:!1}),setTimeout(()=>$(ie=>({...ie,visible:!1})),1800)},ye=Z.useCallback(async ie=>{if(!t||!ie.trim())return;De.current&&De.current.abort();const le=new AbortController;De.current=le,B(!0),V(null),I(null),ne();try{const ze=await ln(`${jt}/api/analyze?id=${encodeURIComponent(ie)}`,{signal:le.signal}),L=await ze.json();if(!ze.ok)throw new Error(L.error||"Erreur serveur");oe(),I(L),z(null),ln(`${jt}/api/star_info?target=${encodeURIComponent(L.target)}`).then(Re=>Re.ok?Re.json():null).then(Re=>{var Ge;(Ge=Re==null?void 0:Re.planets)!=null&&Ge.length&&z(Re)}).catch(()=>{}),se(Re=>[{target:L.target,score:L.score,verdict:L.verdict,period_days:L.period_days,mission:L.mission,date:new Date().toISOString()},...Re].slice(0,50))}catch(ze){if(ze.name==="AbortError"){oe(),B(!1);return}if(ze.message==="Session expirée"||ze.message==="Non authentifié"){Ll(),e(null);return}V(ze.message),oe()}B(!1)},[t]);Z.useEffect(()=>{t&&!P&&ye("Kepler-10")},[t]),Z.useEffect(()=>()=>{De.current&&De.current.abort(),(Oe.current||[]).forEach(clearTimeout)},[]);const Le=ie=>{v(ie),m(ie),ye(ie)},ht=ie=>{i("analysis"),v(ie),m(ie),ye(ie)};if(!t)return l.jsx(Q3,{onLogin:D});const ve=[{key:"analysis",label:"Analyse",icon:ur},{key:"comparison",label:"Comparaison",icon:Ex},{key:"metrics",label:"Métriques",icon:bx},{key:"catalog",label:"Catalogue",icon:Yp},{key:"docs",label:"Documentation",icon:Kp},{key:"profile",label:"Profil",icon:Ys}];return l.jsx(xa.Provider,{value:r,children:l.jsxs("div",{style:{minHeight:"100vh",background:"linear-gradient(165deg,#030510 0%,#060a14 30%,#0c1222 60%,#0d1030 100%)",fontFamily:"'DM Mono','JetBrains Mono',monospace",color:"#e4e8f7",position:"relative",filter:p?"invert(1) hue-rotate(180deg)":"none",transition:"filter 0.5s ease"},children:[l.jsx("style",{children:Px}),l.jsx(Dx,{}),o&&l.jsx(lw,{step:c,onNext:g,onSkip:x}),l.jsxs("header",{style:{position:"relative",zIndex:10,padding:"18px 32px 0",display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:10},children:[l.jsxs("div",{children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:10,marginBottom:4},children:[l.jsxs("div",{style:{width:34,height:34,borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",position:"relative",background:"linear-gradient(135deg,rgba(91,141,239,0.2),rgba(124,58,237,0.2))",border:"1px solid rgba(91,141,239,0.2)",overflow:"hidden"},children:[l.jsx(ur,{size:16,style:{color:"#5b8def",position:"relative",zIndex:1}}),l.jsx("div",{style:{position:"absolute",width:4,height:4,borderRadius:"50%",background:"#f0c040",boxShadow:"0 0 6px rgba(240,192,64,0.5)",animation:"mini-orbit 3s linear infinite"}})]}),l.jsx("h1",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:21,fontWeight:700,background:"linear-gradient(135deg,#5b8def,#7c3aed)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"},children:"ExoPlanet AI"}),l.jsxs("div",{style:{display:"flex",gap:4},children:[l.jsx("span",{style:{fontSize:7,padding:"2px 6px",borderRadius:4,fontWeight:600,background:"rgba(91,141,239,0.1)",color:"#5b8def",border:"1px solid rgba(91,141,239,0.2)",textTransform:"uppercase",letterSpacing:1,fontFamily:"'DM Mono',monospace"},children:"Kepler"}),l.jsx("span",{style:{fontSize:7,padding:"2px 6px",borderRadius:4,fontWeight:600,background:"rgba(232,121,168,0.1)",color:"#e879a8",border:"1px solid rgba(232,121,168,0.2)",textTransform:"uppercase",letterSpacing:1,fontFamily:"'DM Mono',monospace"},children:"TESS"})]})]}),l.jsx("p",{style:{fontSize:11,color:"rgba(160,180,220,0.38)"},children:"Pipeline de detection par transit photometrique · XGBoost + TSFRESH"})]}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"},children:[l.jsx(q3,{status:ae}),l.jsx("button",{onClick:()=>{a(!0),u(0)},title:"Tutoriel interactif",style:{width:28,height:28,borderRadius:"50%",fontSize:12,fontWeight:700,cursor:"pointer",background:"rgba(91,141,239,0.08)",border:"1px solid rgba(91,141,239,0.2)",color:"rgba(91,141,239,0.6)",display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"'Space Grotesk',sans-serif",flexShrink:0},children:"?"}),l.jsxs("div",{"data-tour":"mode-toggle",style:{display:"flex",borderRadius:20,overflow:"hidden",border:"1px solid rgba(91,141,239,0.15)",background:"rgba(10,12,22,0.7)"},children:[l.jsx("button",{onClick:()=>s(!0),style:{padding:"5px 13px",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer",border:"none",transition:"all .2s",background:r?"rgba(124,58,237,0.22)":"transparent",color:r?"#a78bfa":"rgba(160,180,220,0.3)"},children:"✨ Débutant"}),l.jsx("div",{style:{width:1,background:"rgba(91,141,239,0.12)",flexShrink:0}}),l.jsx("button",{onClick:()=>s(!1),style:{padding:"5px 13px",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer",border:"none",transition:"all .2s",background:r?"transparent":"rgba(91,141,239,0.18)",color:r?"rgba(160,180,220,0.3)":"#5b8def"},children:"🔭 Expert"})]}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",borderRadius:7,background:"rgba(91,141,239,0.06)",border:"1px solid rgba(91,141,239,0.1)",cursor:"pointer"},onClick:()=>i("profile"),title:"Voir mon profil",children:[(()=>{var le;const ie=((le=Bf.find(ze=>ze.id===(t==null?void 0:t.avatar)))==null?void 0:le.icon)||Ys;return l.jsx(ie,{size:12,style:{color:"#34d399"}})})(),l.jsx("span",{style:{fontSize:11,color:"#e4e8f7",fontWeight:500},children:t.username})]})]})]}),l.jsx("nav",{"data-tour":"nav",style:{position:"relative",zIndex:10,padding:"14px 32px 0",display:"flex",gap:2,borderBottom:"1px solid rgba(91,141,239,0.07)",paddingBottom:0,marginBottom:0},children:ve.map(({key:ie,label:le,icon:ze})=>l.jsxs("button",{"data-tour":`tab-${ie}`,onClick:()=>i(ie),style:{position:"relative",display:"flex",alignItems:"center",gap:6,padding:"9px 16px",fontSize:11,fontFamily:"'DM Mono',monospace",border:"none",cursor:"pointer",borderBottom:n===ie?"2px solid #5b8def":"2px solid transparent",background:"transparent",color:n===ie?"#5b8def":"rgba(160,180,220,0.4)",transition:"color .2s, border-color .2s"},children:[l.jsx(ze,{size:13}),le,ie==="profile"&&Q.length>0&&l.jsx("span",{style:{fontSize:8,minWidth:14,height:14,display:"flex",alignItems:"center",justifyContent:"center",padding:"0 3px",borderRadius:7,background:"rgba(91,141,239,0.2)",color:"#5b8def",fontWeight:700,fontFamily:"'DM Mono',monospace"},children:Q.length}),n===ie&&l.jsx("div",{style:{position:"absolute",bottom:-2,left:"50%",transform:"translateX(-50%)",width:"60%",height:2,background:"#5b8def",boxShadow:"0 0 8px rgba(91,141,239,0.4),0 0 20px rgba(91,141,239,0.2)",borderRadius:1}})]},ie))}),l.jsxs("main",{style:{position:"relative",zIndex:10,padding:"16px 32px 32px",display:"flex",flexDirection:"column",gap:14},children:[n==="analysis"&&l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"1fr 190px",gap:16,alignItems:"start"},children:[l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14},children:[l.jsxs("div",{style:{position:"relative"},ref:C,children:[l.jsxs("form",{"data-tour":"search",onSubmit:ie=>{if(ie.preventDefault(),A(!1),E>=0&&_[E]){const le=_[E];v(le),m(le),ye(le),y(-1)}else M.trim()&&(m(M.trim()),ye(M.trim()))},style:{display:"flex",alignItems:"center",background:"rgba(15,18,30,0.8)",border:"1px solid rgba(91,141,239,0.14)",borderRadius:11,overflow:"hidden"},children:[l.jsx(Br,{size:13,style:{color:"rgba(91,141,239,0.4)",marginLeft:11}}),l.jsx("input",{value:M,onChange:ie=>{const le=ie.target.value;if(v(le),y(-1),le.length>=2){const ze=le.toLowerCase(),L=Jp.filter($e=>$e.toLowerCase().startsWith(ze)),Re=le.toLowerCase().startsWith("kic")?Uc.filter($e=>$e.toLowerCase().includes(ze)).slice(0,4):[],Ge=[...new Set([...L,...Re])].slice(0,7);b(Ge),A(Ge.length>0)}else b([]),A(!1)},onKeyDown:ie=>{w&&(ie.key==="ArrowDown"?(ie.preventDefault(),y(le=>Math.min(le+1,_.length-1))):ie.key==="ArrowUp"?(ie.preventDefault(),y(le=>Math.max(le-1,-1))):ie.key==="Escape"&&(A(!1),y(-1)))},onBlur:()=>setTimeout(()=>A(!1),150),onFocus:()=>{_.length>0&&A(!0)},placeholder:r?"Nom d'une étoile (ex: Kepler-10)…":"Kepler-10, KIC 11446443, TIC 12345678…",style:{flex:1,padding:"9px 11px",background:"transparent",border:"none",outline:"none",color:"#e4e8f7",fontFamily:"'DM Mono',monospace",fontSize:13}}),l.jsxs("button",{type:"submit",disabled:F,style:{padding:"8px 14px",background:"linear-gradient(135deg,rgba(91,141,239,0.2),rgba(124,58,237,0.2))",border:"none",borderLeft:"1px solid rgba(91,141,239,0.12)",color:"#5b8def",fontFamily:"'DM Mono',monospace",fontSize:11,cursor:"pointer",display:"flex",alignItems:"center",gap:4},children:[F?l.jsx(mi,{size:12,style:{animation:"spin 1s linear infinite"}}):l.jsx(ra,{size:12})," ",r?"Analyser !":"Analyser"]})]}),w&&_.length>0&&l.jsx("div",{style:{position:"absolute",top:"calc(100% + 4px)",left:0,right:0,background:"rgba(8,11,22,0.97)",border:"1px solid rgba(91,141,239,0.2)",borderRadius:9,overflow:"hidden",zIndex:200,boxShadow:"0 8px 24px rgba(0,0,0,0.5)"},children:_.map((ie,le)=>l.jsxs("div",{onMouseDown:()=>{v(ie),m(ie),ye(ie),A(!1),y(-1)},style:{padding:"8px 12px",cursor:"pointer",fontSize:12,fontFamily:"'DM Mono',monospace",background:E===le?"rgba(91,141,239,0.12)":"transparent",color:E===le?"#5b8def":"rgba(200,215,240,0.75)",display:"flex",alignItems:"center",gap:8,borderBottom:le<_.length-1?"1px solid rgba(91,141,239,0.06)":"none",transition:"background .1s"},onMouseEnter:()=>y(le),onMouseLeave:()=>y(-1),children:[l.jsx(Br,{size:10,style:{opacity:.4,flexShrink:0}}),ie]},ie))})]}),W&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,padding:"8px 13px",borderRadius:9,background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",fontSize:12,color:"#f87171"},children:[l.jsx(hr,{size:12}),r?"Étoile introuvable. Essayez un autre nom.":W]}),l.jsx(H3,{progress:j}),r&&P&&!F&&(()=>{const ie=P.score>=.7,le=P.score>=.35,ze=ie?"🌍":le?"🔶":"⭐",L=ie?`${P.target} a probablement une planete !`:le?`${P.target} — resultat ambigu`:`${P.target} — aucune planete detectee`,Re=ie?`Notre intelligence artificielle est confiante a ${Math.round(P.score*100)}%. Un objet en orbite cree des mini-eclipses regulieres visibles sur le graphique ci-dessous.`:le?`La confiance est de ${Math.round(P.score*100)}%. Le signal est present mais peu clair — il faudrait plus de donnees pour conclure.`:`Confiance : ${Math.round(P.score*100)}%. La luminosite de cette etoile ne montre pas de passage regulier d'une planete.`,Ge=ie?"#34d399":le?"#fbbf24":"#94a3b8",$e=ie?"rgba(52,211,153,0.06)":le?"rgba(251,191,36,0.07)":"rgba(148,163,184,0.05)",Te=ie?"rgba(52,211,153,0.25)":le?"rgba(251,191,36,0.25)":"rgba(148,163,184,0.15)";return l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"},children:[l.jsxs(Qe,{style:{padding:"24px 28px",background:$e,border:`1px solid ${Te}`,position:"relative",overflow:"hidden"},children:[l.jsx("div",{style:{position:"absolute",top:0,left:0,width:3,height:"100%",background:ie?"#34d399":le?"#fbbf24":"#64748b",borderRadius:"3px 0 0 3px"}}),l.jsxs("div",{style:{display:"flex",alignItems:"flex-start",gap:16},children:[l.jsx("div",{style:{fontSize:48,lineHeight:1,flexShrink:0},children:ze}),l.jsxs("div",{children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8,marginBottom:6},children:[l.jsx("h2",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:18,fontWeight:700,color:Ge},children:L}),P.mission&&l.jsx("span",{style:{fontSize:8,padding:"2px 7px",borderRadius:4,fontWeight:600,background:P.mission==="Kepler"?"rgba(91,141,239,0.1)":"rgba(232,121,168,0.1)",color:P.mission==="Kepler"?"#5b8def":"#e879a8",border:`1px solid ${P.mission==="Kepler"?"rgba(91,141,239,0.2)":"rgba(232,121,168,0.2)"}`,textTransform:"uppercase",letterSpacing:1,fontFamily:"'DM Mono',monospace"},children:P.mission})]}),l.jsx("p",{style:{fontSize:13,color:"rgba(200,215,240,0.7)",lineHeight:1.6,maxWidth:520},children:Re})]})]})]}),l.jsxs(Qe,{glow:!0,style:{padding:16},children:[l.jsx("h3",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600,marginBottom:4},children:"Ce que voit le télescope"}),l.jsxs("p",{style:{fontSize:11,color:"rgba(160,180,220,0.45)",marginBottom:12},children:["Chaque petit creux dans ce graphique correspond à une planète passant devant l'étoile et bloquant une infime partie de sa lumière.",P.period_days&&` Ce phénomène se répète tous les ${P.period_days} jours.`]}),l.jsx("div",{style:{height:300,borderRadius:10,overflow:"hidden"},children:l.jsx(dc,{data:P.data||[],score:P.score,isLoading:!1})})]}),P.characterization&&l.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10},children:[{label:"Duree d'une orbite",value:P.period_days?`${P.period_days} jours`:"—",iconEl:l.jsx(ga,{size:20,style:{color:"#5b8def"}})},{label:"Taille estimee",value:P.characterization.planet_radius_earth?`${P.characterization.planet_radius_earth} x Terre`:"—",iconEl:l.jsx(Zp,{size:20,style:{color:"#34d399"}})},{label:"Type de planete",value:P.characterization.planet_type||"Indetermine",iconEl:l.jsx(sa,{size:20,style:{color:"#f0c040"}})}].map(({label:R,value:S,iconEl:k})=>l.jsxs(Qe,{style:{padding:"16px 16px",textAlign:"center",border:"1px solid rgba(91,141,239,0.08)",transition:"border-color 0.2s,box-shadow 0.2s"},children:[l.jsx("div",{style:{marginBottom:8,display:"flex",justifyContent:"center"},children:k}),l.jsx("div",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:5,fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:.5},children:R}),l.jsx("div",{style:{fontSize:14,fontWeight:600,color:"#e4e8f7",fontFamily:"'Space Grotesk',sans-serif"},children:S})]},R))})]})})(),!r&&l.jsxs(l.Fragment,{children:[P&&!F&&l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:12,padding:"11px 16px",borderRadius:10,animation:"fadeIn .4s ease-out",position:"relative",overflow:"hidden",background:P.score>=.7?"rgba(52,211,153,0.06)":P.score>=.35?"rgba(251,191,36,0.06)":"rgba(248,113,113,0.06)",border:`1px solid ${P.score>=.7?"rgba(52,211,153,0.2)":P.score>=.35?"rgba(251,191,36,0.2)":"rgba(248,113,113,0.2)"}`},children:[l.jsx("div",{style:{position:"absolute",top:0,left:0,width:3,height:"100%",background:P.score>=.7?"#34d399":P.score>=.35?"#fbbf24":"#f87171",borderRadius:"3px 0 0 3px"}}),P.score>=.7?l.jsx(ma,{size:16,style:{color:"#34d399"}}):P.score>=.35?l.jsx(Lc,{size:16,style:{color:"#fbbf24"}}):l.jsx(hr,{size:16,style:{color:"#f87171"}}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:8},children:[l.jsx("span",{style:{fontSize:13,fontWeight:600,color:"#e4e8f7"},children:P.target}),l.jsx("span",{style:{fontSize:7,padding:"2px 6px",borderRadius:4,fontWeight:600,background:P.mission==="Kepler"?"rgba(91,141,239,0.1)":"rgba(232,121,168,0.1)",color:P.mission==="Kepler"?"#5b8def":"#e879a8",border:`1px solid ${P.mission==="Kepler"?"rgba(91,141,239,0.2)":"rgba(232,121,168,0.2)"}`,textTransform:"uppercase",letterSpacing:1,fontFamily:"'DM Mono',monospace"},children:P.mission}),l.jsx("span",{style:{fontSize:12,color:"rgba(160,180,220,0.55)"},children:P.verdict})]}),l.jsxs("div",{style:{marginLeft:"auto",fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",display:"flex",alignItems:"center",gap:6},children:[l.jsx(ur,{size:10,style:{opacity:.5}})," ",P.analyzed_by]})]}),l.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .6s ease-out"},children:[l.jsxs(Qe,{glow:!0,style:{padding:14},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10},children:[l.jsxs("div",{children:[l.jsx("h2",{style:{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600},children:"Courbe de Lumière Repliée"}),l.jsx("p",{style:{fontSize:10,color:"rgba(160,180,220,0.38)",marginTop:1},children:P?`${P.target} — P = ${P.period_days} j`:"En attente d'une analyse…"})]}),l.jsxs("button",{onClick:()=>ye(d),disabled:F,style:{display:"flex",alignItems:"center",gap:4,padding:"4px 8px",borderRadius:6,background:"rgba(91,141,239,0.07)",border:"1px solid rgba(91,141,239,0.14)",color:"#5b8def",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer"},children:[l.jsx(Cx,{size:11})," Recharger"]})]}),l.jsx("div",{style:{height:340,borderRadius:10,overflow:"hidden"},children:l.jsx(dc,{data:(P==null?void 0:P.data)||[],score:(P==null?void 0:P.score)||.5,isLoading:F})})]}),l.jsxs("div",{style:{display:"grid",gridTemplateColumns:"280px 1fr 1fr",gap:14},children:[l.jsxs(Qe,{style:{display:"flex",flexDirection:"column",alignItems:"center",padding:"16px 14px"},children:[l.jsx("h3",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:8,textTransform:"uppercase",letterSpacing:1.5},children:"Verdict de l'IA"}),P?l.jsx(eh,{score:P.score,scoreStd:P.score_std??null}):l.jsx("div",{style:{color:"rgba(160,180,220,0.3)",fontSize:12,padding:16},children:"En attente…"})]}),((Ve=P==null?void 0:P.feature_importances)==null?void 0:Ve.length)>0?l.jsx(Qe,{style:{padding:14},children:l.jsx(X3,{features:P.feature_importances})}):l.jsx("div",{}),P?l.jsxs(Qe,{style:{padding:14},children:[l.jsx("h3",{style:{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:10,textTransform:"uppercase",letterSpacing:1.5},children:"Caractéristiques"}),l.jsx(Lx,{data:P})]}):l.jsx("div",{})]}),l.jsx(Qe,{glow:!0,style:{padding:0,overflow:"hidden"},children:l.jsx("div",{style:{height:380,borderRadius:14},children:l.jsx(BT,{data:P,nasaPlanets:G==null?void 0:G.planets})})})]}),P&&l.jsx(G3,{data:P}),P&&l.jsx(W3,{target:d})]})]}),l.jsx(ow,{current:d,onPick:Le})]}),n==="comparison"&&l.jsx(aw,{}),n==="metrics"&&l.jsx(Z3,{}),n==="catalog"&&l.jsx(J3,{onAnalyze:ht}),n==="docs"&&l.jsx(Nx,{}),n==="profile"&&l.jsx(cw,{authState:t,history:Q,onLogout:q,setAuthState:e,isLightMode:p,setIsLightMode:h,onAnalyze:ht,onClearHistory:async()=>{await ln(`${jt}/api/history`,{method:"DELETE"}),se([])},onDeleteHistory:async ie=>{await ln(`${jt}/api/history/${ie}`,{method:"DELETE"}),se(le=>le.filter((ze,L)=>L!==ie))}}),l.jsxs("div",{style:{borderTop:"1px solid rgba(91,141,239,0.06)",padding:"16px 0 8px",marginTop:8},children:[l.jsxs("div",{style:{display:"flex",justifyContent:"center",alignItems:"center",gap:12,marginBottom:8},children:[l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:5},children:[l.jsx("div",{style:{width:6,height:6,borderRadius:"50%",background:"#5b8def",boxShadow:"0 0 6px rgba(91,141,239,0.4)"}}),l.jsx("span",{style:{fontSize:9,color:"rgba(91,141,239,0.6)",fontFamily:"'DM Mono',monospace",letterSpacing:.5,textTransform:"uppercase"},children:"Kepler"})]}),l.jsx("div",{style:{width:1,height:10,background:"rgba(91,141,239,0.1)"}}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:5},children:[l.jsx("div",{style:{width:6,height:6,borderRadius:"50%",background:"#e879a8",boxShadow:"0 0 6px rgba(232,121,168,0.4)"}}),l.jsx("span",{style:{fontSize:9,color:"rgba(232,121,168,0.6)",fontFamily:"'DM Mono',monospace",letterSpacing:.5,textTransform:"uppercase"},children:"TESS"})]}),l.jsx("div",{style:{width:1,height:10,background:"rgba(91,141,239,0.1)"}}),l.jsxs("div",{style:{display:"flex",alignItems:"center",gap:5},children:[l.jsx("div",{style:{width:6,height:6,borderRadius:"50%",background:"#34d399",boxShadow:"0 0 6px rgba(52,211,153,0.4)"}}),l.jsx("span",{style:{fontSize:9,color:"rgba(52,211,153,0.5)",fontFamily:"'DM Mono',monospace",letterSpacing:.5,textTransform:"uppercase"},children:"NASA MAST"})]}),l.jsx("div",{style:{width:1,height:10,background:"rgba(91,141,239,0.1)"}}),l.jsx("span",{style:{fontSize:9,color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",letterSpacing:.5},children:"Transit Photometrique · XGBoost"})]}),l.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:9,color:"rgba(160,180,220,0.18)"},children:[l.jsx("span",{children:"ECE Paris — ING4 Group 1 · S. Gallais, M. Rolland, C. De Blauwe, M. Leitao, O. Schwartz, K. Benjelloum"}),l.jsx("span",{style:{fontFamily:"'DM Mono',monospace"},children:"5000+ exoplanetes confirmees"})]})]})]})]})})}qu.createRoot(document.getElementById("root")).render(l.jsx(sv.StrictMode,{children:l.jsx(uw,{})}));
