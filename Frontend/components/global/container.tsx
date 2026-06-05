"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface Props {
    className?: string;
    children: React.ReactNode;
    delay?: number;
    reverse?: boolean;
    simple?: boolean;
    eager?: boolean;
}

const Container = ({ children, className, delay = 0.2, reverse, simple, eager }: Props) => {
    return (
        <motion.div
            className={cn("w-full h-full", className)}
            initial={{ opacity: 0, y: reverse ? -20 : 20 }}
            {...(eager
                ? { animate: { opacity: 1, y: 0 } }
                : { whileInView: { opacity: 1, y: 0 }, viewport: { once: true } }
            )}
            transition={{
                delay,
                duration: simple ? 0.2 : 0.4,
                type: simple ? "keyframes" : "spring",
                stiffness: simple ? 100 : undefined,
            }}
        >
            {children}
        </motion.div>
    );
};

export default Container;
