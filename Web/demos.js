"use strict";
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const fixed = value => value.toLocaleString("kk-KZ", {maximumFractionDigits: 4});
  document.querySelectorAll("[data-ml-demo]").forEach(demo => {
    const input = demo.querySelector("input");
    const svg = demo.querySelector("svg");
    const output = demo.querySelector("output");
    const detail = demo.querySelector(".demo-detail");
    const kind = demo.dataset.mlDemo;
    function shape(tag, attributes, label) {
      const node = document.createElementNS(NS, tag);
      Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key, value));
      if (label !== undefined) node.textContent = label;
      svg.append(node);
      return node;
    }
    const text = (x,y,label) => shape("text", {x,y,fill:"currentColor","font-size":15}, label);
    const line = (x1,y1,x2,y2,color="currentColor") => shape("line",{x1,y1,x2,y2,stroke:color,"stroke-width":2});
    const circle = (x,y,color,r=6) => shape("circle",{cx:x,cy:y,r,fill:color,stroke:"currentColor","stroke-width":1});
    const blue = "#2587bc", teal = "#12a893", amber = "#ce8237";
    const path = (points, color=teal) => shape("polyline",{points:points.map(p=>p.join(",")).join(" "),fill:"none",stroke:color,"stroke-width":3});
    function update() {
      const value = Number(input.value);
      svg.replaceChildren();
      detail.textContent = "";
      let state = {};
      if (kind === "logistic") {
        const px = x => 45 + (x+6)/12*510, py = p => 220-p*185;
        line(45,220,555,220); line(45,220,45,25);
        line(45,py(.5),555,py(.5),amber);
        path(Array.from({length:121},(_,i)=>{const x=-6+i/10;return [px(x),py(sigmoid(x))];}));
        const p = sigmoid(value);
        circle(px(value),py(p),blue,8);
        text(45,250,"−6"); text(537,250,"6"); text(8,40,"1"); text(8,135,"0,5");
        output.textContent = "x = "+fixed(value)+" · p = "+fixed(p)+" · Болжам класы: "+Number(p>=.5);
        detail.textContent = "Сызық — 0,5 шешім шегі. Ықтималдық үздіксіз өзгереді, ал класс шектен өткенде өзгереді.";
        state = {x:value,probability:p,prediction:Number(p>=.5)};
      } else if (kind === "gradient") {
        let weight=2;
        const weights=[weight];
        for(let i=0;i<12;i++){weight*=(1-value);weights.push(weight);}
        const scale=Math.max(2,...weights.map(Math.abs));
        const py=w=>130-w/scale*90;
        line(45,130,555,130); line(45,30,45,225);
        path(weights.map((w,i)=>[45+i*42,py(w)]));
        weights.forEach((w,i)=>circle(45+i*42,py(w),blue,4));
        text(45,250,"0-қадам"); text(465,250,"12-қадам"); text(8,135,"0");
        output.textContent = "η = "+fixed(value)+" · w₁₂ = "+fixed(weight)+" · L(w₁₂) = "+fixed(weight*weight/2);
        detail.textContent = value===0 ? "Қадам нөлге тең: параметр өзгермейді." : value<2 ? "Осы квадраттық функцияда 0 < η < 2 кезінде параметр минимумға жақындайды." : value===2 ? "η = 2 кезінде таңба алмасады, бірақ қате азаймайды." : "η > 2 кезінде ауытқу ұлғаяды. Тік шкала траекторияға бейімделеді.";
        state={rate:value,weight,loss:weight*weight/2};
      } else if (kind === "tree") {
        const labels=[0,0,1,0,1,1];
        const gini=values=>{const p=values.reduce((a,b)=>a+b,0)/values.length;return 2*p*(1-p);};
        const left=labels.filter((_,i)=>i+1<=value),right=labels.filter((_,i)=>i+1>value);
        const mixed=(left.length*gini(left)+right.length*gini(right))/6;
        line(50,150,550,150);
        labels.forEach((label,i)=>{const x=75+i*90;circle(x,125,label?teal:blue,13);text(x-5,180,String(i+1));text(x-5,95,String(label));});
        const split=75+(value-1)*90;line(split,65,split,200,amber);
        text(50,235,"Көк: 0 класы · жасыл: 1 класы");
        output.textContent="t = "+fixed(value)+" · Салмақталған Джини: "+fixed(mixed)+" · Кемуі: "+fixed(.5-mixed);
        detail.textContent="Сол тармақ: "+left.join(", ")+". Оң тармақ: "+right.join(", ")+". Салмақтар тармақ көлемдеріне тең.";
        state={threshold:value,gini:mixed,gain:.5-mixed};
      } else if (kind === "metrics") {
        const scores=[.1,.3,.4,.6,.8,.9],labels=[0,1,0,1,0,1];
        let tp=0,fp=0,tn=0,fn=0;
        scores.forEach((score,i)=>{const predicted=score>=value;if(predicted&&labels[i])tp++;else if(predicted)fp++;else if(labels[i])fn++;else tn++;});
        const precision=tp+fp?tp/(tp+fp):null,recall=tp/(tp+fn),f1=2*tp+fp+fn?2*tp/(2*tp+fp+fn):null;
        const cells=[["TN",tn,65,55],["FP",fp,310,55],["FN",fn,65,145],["TP",tp,310,145]];
        cells.forEach(([name,count,x,y])=>{shape("rect",{x,y,width:225,height:75,rx:8,fill:name==="TP"||name==="TN"?teal:amber,opacity:.18});text(x+25,y+43,name+" = "+count);});
        text(85,25,"Болжам 0");text(330,25,"Болжам 1");text(15,100,"0");text(15,190,"1");
        const fmt=v=>v===null?"анықталмаған":fixed(v);
        output.textContent="Шек: "+fixed(value)+" · Accuracy: "+fixed((tp+tn)/6)+" · Precision: "+fmt(precision)+" · Recall: "+fixed(recall)+" · F1: "+fmt(f1);
        detail.textContent="TN — дұрыс теріс, FP — жалған оң, FN — жалған теріс, TP — дұрыс оң. Бағандар: болжам 0 және 1; жолдар: нақты 0 және 1.";
        state={threshold:value,tp,fp,tn,fn,precision,recall,f1,accuracy:(tp+tn)/6};
      } else if (kind === "regularization") {
        const weight=60/(30+value), mse=[1,2,3,4].reduce((sum,x)=>sum+(2*x-weight*x)**2,0)/4;
        const px=x=>45+x*120,py=y=>225-y*23;
        line(45,225,550,225);line(45,225,45,25);
        path([[px(0),py(0)],[px(4),py(weight*4)]]);
        [1,2,3,4].forEach(x=>circle(px(x),py(2*x),blue));
        text(45,250,"0");text(520,250,"x = 4");text(8,42,"8");
        output.textContent="λ = "+fixed(value)+" · w = "+fixed(weight)+" · Оқыту MSE: "+fixed(mse);
        detail.textContent="Көк нүктелер — берілген деректер. Жасыл сызық — Ridge болжамы. Қатені тек осы төрт оқыту нүктесінде есептейміз; бұл жалпылау бағасы емес.";
        state={lambda:value,weight,mse};
      } else if (kind === "kmeans") {
        const points=[0,2,8,10],centers=[value,8];
        const groups=points.map(x=>Math.abs(x-centers[0])<=Math.abs(x-centers[1])?0:1);
        const next=centers.map((_,group)=>{const kept=points.filter((__,i)=>groups[i]===group);return kept.reduce((a,b)=>a+b,0)/kept.length;});
        const px=x=>55+(x+6)*30;
        line(45,120,555,120);line(45,200,555,200);
        points.forEach((x,i)=>{circle(px(x),120,groups[i]?teal:blue,10);text(px(x)-5,95,String(x));});
        centers.forEach((x,i)=>{line(px(x),40,px(x),70,i?teal:blue);text(px(x)-12,30,"c"+(i+1));line(px(x),70,px(next[i]),180,amber);circle(px(next[i]),200,i?teal:blue,12);});
        text(45,250,"Төменгі қатар — жаңартылған орталықтар");
        const before=points.reduce((s,x,i)=>s+(x-centers[groups[i]])**2,0),after=points.reduce((s,x,i)=>s+(x-next[groups[i]])**2,0);
        output.textContent="Орталықтар: ["+centers.map(fixed).join("; ")+"] → ["+next.map(fixed).join("; ")+"] · SSE: "+fixed(before)+" → "+fixed(after);
        state={centers,next,groups,before,after};
      } else if (kind === "neural") {
        const z=2*value-1,hidden=Math.max(0,z),p=sigmoid(hidden-1);
        line(105,125,295,125,blue);line(315,125,505,125,teal);
        [[95,"x = 2",blue],[305,"ReLU",teal],[515,"sigmoid",amber]].forEach(([x,label,color])=>{circle(x,125,color,35);text(x-30,190,label);});
        text(155,85,"w = "+fixed(value));text(245,35,"b = −1");text(395,85,"h − 1");
        output.textContent="z = "+fixed(z)+" · h = "+fixed(hidden)+" · p = "+fixed(p);
        detail.textContent=z<=0?"ReLU теріс кірісті нөлге айналдырады. Осы аймақта h өзгермейді.":"ReLU оң кірісті сақтайды. Шығыс сигмоидасы мәнді 0 мен 1 аралығына түрлендіреді.";
        state={weight:value,z,hidden,probability:p};
      }
      demo.dataset.state=JSON.stringify(state);
    }
    input.addEventListener("input",update);
    update();
  });
})();
