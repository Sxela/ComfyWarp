import { app } from "../../scripts/app.js";

class QueueMany {
  constructor(node) {
    this.serializedCtx = {};
    this.node = node;
    for (const [i, w] of this.node.widgets.entries()) {
      if (w.name === "start") {
        this.start = w;
      }
      if (w.name === "end") {
        this.end = w;
      }
      if (w.name === "current_number") {
        this.current_number = w;
      }
    }

    this.QueueButton = this.node.addWidget(
      "button",
      "Queue from " + this.start.value,
      null,
      () => {
        const ui = window.comfyAPI.app.app.ui;
        ui.autoQueueMode = "instant"; // or whatever mode you want
        ui.autoQueueEnabled = true; // or false to disable
        this.current_number.value = this.start.value;
        if (this.current_number.value <= this.end.value) {
          app.queuePrompt(0, 1);
        }
      }
    );

    this.current_number.serializeValue = async (node, index) => {
      return this.current_number.value;
    };

    this.current_number.afterQueued = () => {
      this.current_number.value += 1;
      for (const [i, w] of this.node.widgets.entries()) {
        if (w.name === "start") {
          this.start = w;
        }
        if (w.name === "end") {
          this.end = w;
        }
        if (w.name === "current_number") {
          this.current_number = w;
        }
      }
      if (this.current_number.value > this.end.value) {
        const ui = window.comfyAPI.app.app.ui;
        ui.autoQueueMode = "disabled";
        ui.autoQueueEnabled = false;
      }
    };

  }
}

app.registerExtension({
  name: "warpfusion.queuemany",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name === "FixedQueue") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
        this.queueMany = new QueueMany(this);
      };
    }
  },
});
