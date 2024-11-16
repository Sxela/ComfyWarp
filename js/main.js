import { app } from "../../scripts/app.js";

class QueueMany {
    constructor(node) {
        this.serializedCtx = {};
        this.node = node;
        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === 'start') {
                this.start = w;
            }
            if (w.name === 'end') {
                this.end = w;
            }
            if (w.name === 'current_number') {
                this.current_number = w;
            }
        }
        this.current_number.value = this.start.value
        console.log(this.start, this.end, this.current_number)

        this.QueueButton = this.node.addWidget("button", 'Queue Many', null, () => {
            console.log('lolo')
            this.current_number.value = this.start.value
            for (let i=this.start.value; i<=this.end.value; i++){
                console.log('iiii', i)
			    app.queuePrompt(0, 1)
            
            }
        })

        this.current_number.serializeValue = async (node, index) => {return this.current_number.value}

        this.current_number.afterQueued = () => {
            this.current_number.value += 1
            if (this.current_number.value > this.end.value) {
                this.current_number.value = this.start.value
            }
        }
    }
}

app.registerExtension({
    name: "warpfusion.queuemany",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        // console.log(nodeData.name)
        if (nodeData.name === "FixedQueue"){
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.queueMany = new QueueMany(this);
        }
    };
    }})

