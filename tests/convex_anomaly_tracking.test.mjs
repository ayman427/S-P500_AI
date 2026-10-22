import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {register,capture,settle,checkpoint} from "../convex/anomalyTracking.js";
const secret="tracking-test-secret";
process.env.PREDICTION_STORE_SECRET=secret;
const hash=text => createHash("sha256").update(text).digest("hex");
const settings={tickers:["AAPL"],method:"price_volume_observations_v1",horizonSessions:20,declineThreshold:-.1,multiSignalMinimum:2,volumeThreshold:2,marketThreshold:.03,accelerationThreshold:.05};
const spec=JSON.stringify(settings);
const key="price_volume_prospective_v1";
function context() {
  const tables={anomalyProtocols:[],anomalyObservations:[],anomalyTrackingChecks:[]};
  return {tables,db:{query:table => ({withIndex:(_name,select) => {
    const filters=[];const q={eq:(f,v)=>{filters.push(r=>r[f]===v);return q;},lte:(f,v)=>{filters.push(r=>r[f]<=v);return q;}};select(q);
    return {unique:async()=>tables[table].find(r=>filters.every(f=>f(r)))??null};
  }}),insert:async(table,row)=>{const id=table+tables[table].length;tables[table].push({...row,_id:id});return id;},
  patch:async(id,updates)=>{const row=Object.values(tables).flat().find(r=>r._id===id);Object.assign(row,updates);}}};
}
function weekdays(start,end) {
  const rows=[];for(let d=new Date(start);d<=new Date(end);d.setUTCDate(d.getUTCDate()+1)) {
    if(![0,6].includes(d.getUTCDay())&&d.toISOString().slice(0,10)!=="2024-05-27") rows.push({date:d.toISOString().slice(0,10),close:100,volume:100});
  }return rows;
}
function record() {
  const stock=weekdays("2024-04-01","2024-05-03"); stock.at(-1).close=110;
  const snapshot=JSON.stringify({stock,market:[{date:"2024-05-02",close:100},{date:"2024-05-03",close:101}]});
  return {key:key+":AAPL:2024-05-03",protocol:key,protocolHash:hash(spec),ticker:"AAPL",eventDate:"2024-05-03",
    closeAt:Date.parse("2024-05-03T20:00:00Z"),nextOpenAt:Date.parse("2024-05-06T13:30:00Z"),dueDate:"2024-06-03",dueAt:Date.parse("2024-06-03T20:30:00Z"),eventClose:110,inputSnapshot:snapshot,inputSha256:hash(snapshot),
    signals:[{id:"price",value:.1,threshold:.05,status:"elevated"},{id:"volume",value:2,threshold:2,status:"elevated"},{id:"market",value:.09,threshold:.03,status:"elevated"},{id:"acceleration",value:.02,threshold:.05,status:"within_rule"}],flagged:true};
}
async function frozen(ctx,t) {
  t.mock.method(Date,"now",()=>Date.parse("2024-05-02T00:00:00Z"));
  await register._handler(ctx,{secret,key,hash:hash(spec),spec});
  t.mock.restoreAll();t.mock.method(Date,"now",()=>Date.parse("2024-05-03T21:00:00Z"));
}
test("service authentication and immutable protocol",async t=>{
  const ctx=context();await assert.rejects(register._handler(ctx,{secret:"wrong",key,hash:hash(spec),spec}),/Unauthorized/);
  await frozen(ctx,t);
  const changed=JSON.stringify({...settings,tickers:["MSFT"]});
  await assert.rejects(register._handler(ctx,{secret,key,hash:hash(changed),spec:changed}),/Frozen protocol mismatch/);
});
test("first observation remains immutable and outcomes cannot settle early",async t=>{
  const ctx=context();await frozen(ctx,t);const r=record();
  assert.equal((await capture._handler(ctx,{secret,record:r})).created,true);
  await capture._handler(ctx,{secret,record:r});assert.equal(ctx.tables.anomalyObservations.length,1);
  await assert.rejects(settle._handler(ctx,{secret,key:r.key,outcomeSnapshot:"[]",outcomeSha256:hash("[]")}),/not due/);
  const bad={...r,inputSha256:"a".repeat(64)};
  await assert.rejects(capture._handler(ctx,{secret,record:bad}),/snapshot hash/);
});
test("late or hindsight observations and changed thresholds are rejected",async t=>{
  const ctx=context();await frozen(ctx,t);const r=record();
  const changed=structuredClone(r);changed.signals[1].threshold=1;
  await assert.rejects(capture._handler(ctx,{secret,record:changed}),/threshold/);
  t.mock.restoreAll();t.mock.method(Date,"now",()=>r.nextOpenAt);
  await assert.rejects(capture._handler(ctx,{secret,record:r}),/late/);
});
test("settlement requires a full hash-verified window and is idempotent",async t=>{
  const ctx=context();await frozen(ctx,t);const r=record();await capture._handler(ctx,{secret,record:r});
  t.mock.restoreAll();t.mock.method(Date,"now",()=>r.dueAt);
  const rows=weekdays("2024-05-03","2024-06-03").map(row=>({date:row.date,close:25}));
  assert.equal(rows.length,21);rows.at(-1).close=20;
  const text=JSON.stringify(rows);await settle._handler(ctx,{secret,key:r.key,outcomeSnapshot:text,outcomeSha256:hash(text)});
  const stored=ctx.tables.anomalyObservations[0];assert.equal(stored.reversal,true);assert.ok(Math.abs(stored.return20+.2)<1e-12);
  assert.equal(stored.eventClose,110);assert.equal(stored.inputSnapshot,r.inputSnapshot);
  assert.equal((await settle._handler(ctx,{secret,key:r.key,outcomeSnapshot:text,outcomeSha256:hash(text)})).settled,false);
});
test("coverage never claims a capture without a saved observation",async t=>{
  const ctx=context();await frozen(ctx,t);
  await assert.rejects(checkpoint._handler(ctx,{secret,protocol:key,ticker:"AAPL",session:"2024-05-03",status:"captured"}),/not confirmed/);
});
