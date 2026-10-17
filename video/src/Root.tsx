import { Composition } from "remotion";
import { END, FPS } from "./timeline";
import { Trailer } from "./Trailer";
import { END2, FPS2, Trailer2 } from "./Trailer2";
import { END3, FPS3, Trailer3 } from "./Trailer3";

export const Root = () => (
  <>
    <Composition id="Trailer" component={Trailer} durationInFrames={END} fps={FPS} width={1920} height={1080} />
    <Composition id="Trailer2" component={Trailer2} durationInFrames={END2} fps={FPS2} width={1920} height={1080} />
    <Composition id="Trailer3" component={Trailer3} durationInFrames={END3} fps={FPS3} width={1920} height={1080} />
  </>
);
